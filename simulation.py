#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 01:56:26 2025

@author: kimberlybond
"""
# Data Structures: The request class definition

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

@dataclass
class Request:
    """
    Represents a single LLM query/job in the simulation.
    """
    # --- Immutable Properties (Defined at Arrival) ---
    arrival_time: float
    """Timestamp when the request entered the system."""

    prompt_length: int
    """Number of tokens in the input prompt (L_i). Fixed for the demo."""

    output_budget: int
    """Total number of tokens to generate (B_i). Fixed for the demo."""

    request_id: int
    """Unique identifier for debugging and tracking."""

    # --- Mutable State (Tracked during Processing) ---
    prefill_completed: bool = False
    """Flag indicating if the compute-heavy prefill phase is finished."""

    tokens_decoded: int = 0
    """Counter for the number of output tokens currently generated."""

    @property
    def remaining_tokens(self) -> int:
        """Returns the number of tokens left to decode."""
        return self.output_budget - self.tokens_decoded

    @property
    def is_completed(self) -> int:
        """Returns True if the request has finished all decoding steps."""
        return self.prefill_completed and (self.tokens_decoded >= self.output_budget)
    

# Service Physics: ServiceTimeModelClass

class ServiceTimeModel:
    """
    Calculates the GPU processing time based on batch token load.
    Implements the logic: S(b) = C + a * max(0, b - b0)
    where C and a are stochastic random variables.

    This class is the "physics" of the simulator: it tells us how long a
    given batch of tokens takes on the GPU.
    """
    def __init__(
        self,
        mean_setup_cost: float,
        mean_marginal_cost: float,
        b0_threshold: int = 0,
        rng=None,
    ):
        """
        Initialize the stochastic service time model.

        Args:
            mean_setup_cost (float): The average fixed cost (C) in seconds.
            mean_marginal_cost (float): The average cost per token (a) in seconds.
            b0_threshold (int): The token count threshold where linear scaling begins.
            rng: Optional NumPy random Generator for reproducible randomness.
                 If None, a new default_rng() is created.
        """
        self.mean_setup_cost = mean_setup_cost
        self.mean_marginal_cost = mean_marginal_cost
        self.b0 = b0_threshold
        # Use an explicit RNG so that all service-time randomness can be
        # controlled from the outside (e.g., by passing a seeded generator).
        self.rng = rng or np.random.default_rng()

    def get_service_time(self, batch_token_load: int) -> float:
        """
        Calculate the time to process a batch with a specific total token load.

        Args:
            batch_token_load (int): Sum of prompt tokens for all prefill jobs in
                                    the batch + 1 token for each decoding job.

        Returns:
            float: The duration of the operation in seconds.
        """
        # 1. Sample stochastic parameters for *this* GPU operation.
        # Using an Exponential distribution makes each C and a random, so
        # TTFT/TBT distributions are not deterministic even for fixed L_i, B_i.
        c_sample = self.rng.exponential(self.mean_setup_cost)
        a_sample = self.rng.exponential(self.mean_marginal_cost)

        # 2. Apply the piecewise linear function
        #    S(b) = C + a * max(0, b - b0)
        excess_load = max(0, batch_token_load - self.b0)
        service_time = c_sample + (a_sample * excess_load)

        return service_time
    
    
# Sceduler A - No Batching

def simulate_run_to_completion(
    num_jobs: int,
    lambda_rate: float,
    prompt_length: int,
    output_budget: int,
    service_model: ServiceTimeModel,
    rng=None,
    log_events: bool = True,
):
    """
    Simulate Scheduler A: Run-to-completion (no batching, single GPU).

    Policy / batching behavior:
    - Jobs are served strictly in FIFO order.
    - For each job, we first run a **prefill** step for the whole prompt
      (batch_token_load = L_i), then run **decode** steps one token at a time
      (batch_token_load = 1) *only for that job*.
    - There is therefore **no cross-job batching** under this policy.

    Returns:
        results dict + 'events': list of event dicts in time order.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1. Generate Poisson arrivals (inter-arrival times ~ Exp(lambda))
    inter_arrivals = rng.exponential(1.0 / lambda_rate, size=num_jobs)
    arrival_times = np.cumsum(inter_arrivals)

    # Event list: append a dict every time something happens
    events = []

    def record_event(time: float, event_type: str, job_id: int, **extra):
        """Helper to record and optionally print events.

        This is purely instrumentation: it does not affect the physics of the
        simulation but lets us reconstruct timelines and compute metrics.
        """
        ev = {"time": time, "type": event_type, "job_id": job_id}
        ev.update(extra)
        events.append(ev)
        #if log_events:
         #   print(f"[{time:8.4f}] {event_type:20s} job={job_id} {extra}")

    # Tracks when the single GPU becomes free; new jobs cannot start before this.
    server_available_time = 0.0

    # Arrays for post-hoc metric computation.
    start_times = np.zeros(num_jobs)
    completion_times = np.zeros(num_jobs)
    ttft = np.zeros(num_jobs)  # Time to first token (end of prefill - arrival)
    tbt = np.zeros(num_jobs)   # Average time between decoded tokens

    for i in range(num_jobs):
        arrival = arrival_times[i]
        record_event(arrival, "ARRIVAL", i)

        # When can this job start?  Either at arrival or when GPU finishes
        # the previous job (run-to-completion, single server).
        start_service = max(arrival, server_available_time)
        start_times[i] = start_service
        record_event(start_service, "SERVICE_START", i)

        # Create the logical Request object for this job.
        req = Request(
            arrival_time=arrival,
            prompt_length=prompt_length,
            output_budget=output_budget,
            request_id=i,
        )

        # --- Prefill Phase -------------------------------------------------
        # Entire prompt is processed as a single batch: batch_token_load = L_i.
        prefill_time = service_model.get_service_time(req.prompt_length)
        prefill_end = start_service + prefill_time
        req.prefill_completed = True

        # TTFT is the delay from arrival until prefill finishes and the
        # first token can be emitted.
        ttft[i] = prefill_end - arrival
        record_event(
            prefill_end,
            "PREFILL_COMPLETE",
            i,
            ttft=ttft[i],
            prefill_time=prefill_time,
        )

        # --- Decode Phase --------------------------------------------------
        # Under Scheduler A, we decode this job's tokens one by one with
        # batch_token_load = 1, so there is no token-level batching either.
        decode_times = []
        current_time = prefill_end
        for step in range(req.output_budget):
            step_time = service_model.get_service_time(batch_token_load=1)
            current_time += step_time
            decode_times.append(step_time)
            req.tokens_decoded += 1

            record_event(
                current_time,
                "DECODE_STEP",
                i,
                step=step + 1,
                step_time=step_time,
                tokens_decoded=req.tokens_decoded,
            )

        total_decode_time = sum(decode_times)
        # Average time between tokens (TBT) for this job.
        tbt[i] = np.mean(decode_times) if decode_times else 0.0

        # --- Completion ----------------------------------------------------
        completion_time = prefill_end + total_decode_time
        completion_times[i] = completion_time
        # Run-to-completion: GPU cannot start the next job until this one ends.
        server_available_time = completion_time

        record_event(
            completion_time,
            "JOB_COMPLETE",
            i,
            total_service_time=prefill_time + total_decode_time,
            tbt=tbt[i],
        )

    total_time = completion_times[-1] - arrival_times[0]
    throughput = num_jobs / total_time

    results = {
        "arrival_times": arrival_times,
        "start_times": start_times,
        "completion_times": completion_times,
        "ttft": ttft,
        "tbt": tbt,
        "throughput": throughput,
        "events": events,   # <--- full event list here
    }
    return results


# Scheduler B - with batching

def simulate_prefill_priority_batching(
    num_jobs: int,
    lambda_rate: float,
    prompt_length: int,
    output_budget: int,
    service_model: ServiceTimeModel,
    max_batch_size: int = 4,  # K in the spec
    rng=None,
    log_events: bool = False,
):
    """
    Scheduler B: **Basic Prefill-Prioritization with Batching**.

    Policy:
    - Single GPU server, same arrival process and request model as Scheduler A.
    - At each decision point (when the GPU becomes free):
      1. **Prefill priority:** If there are any jobs that have arrived but
         not yet done prefill, form a prefill batch of up to `max_batch_size`
         jobs and run ONE batched prefill operation over their prompts.
      2. **Decode otherwise:** If no jobs need prefill, form a decode batch
         over all currently active decoding jobs, giving each job exactly
         one output token in that GPU operation (batch_token_load = #active).

    This implements the spec: "If there are new jobs waiting, form a batch of
    Prefill tasks (up to K). Otherwise, form a batch of Decode tasks for all
    active jobs (one token per job)."
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate arrivals (same Poisson process as Scheduler A)
    inter_arrivals = rng.exponential(1.0 / lambda_rate, size=num_jobs)
    arrival_times = np.cumsum(inter_arrivals)

    # Create all logical Request objects up front
    requests = [
        Request(
            arrival_time=arrival_times[i],
            prompt_length=prompt_length,
            output_budget=output_budget,
            request_id=i,
        )
        for i in range(num_jobs)
    ]

    # Tracking
    events = []
    current_time = 0.0

    # Queues / sets of indices for each state
    prefill_queue = list(range(num_jobs))  # Jobs that still need prefill
    active_decoding = []                   # Jobs that have finished prefill but not all tokens
    completed = []                         # Jobs fully completed

    ttft = np.zeros(num_jobs)
    tbt_times = {i: [] for i in range(num_jobs)}  # Per-job list of inter-token times
    completion_times = np.full(num_jobs, np.nan)

    next_arrival_idx = 0
    arrived = []  # Jobs that have arrived (arrival_time <= current_time)

    def record_event(time, event_type, job_ids, **extra):
        #if log_events:
        #    print(f"[{time:8.4f}] {event_type:20s} jobs={job_ids} {extra}")
        events.append({"time": time, "type": event_type, "jobs": job_ids, **extra})

    while len(completed) < num_jobs:
        # 1. Process all arrivals up to current_time so they become eligible
        while next_arrival_idx < num_jobs and arrival_times[next_arrival_idx] <= current_time:
            arrived.append(next_arrival_idx)
            record_event(arrival_times[next_arrival_idx], "ARRIVAL", [next_arrival_idx])
            next_arrival_idx += 1

        # 2. Decide what to schedule at this decision point.
        # Priority 1: Batch prefills for any arrived jobs that still need prefill.
        available_prefills = [i for i in arrived if i in prefill_queue]

        if available_prefills:
            # Batch up to max_batch_size new prefills
            batch = available_prefills[:max_batch_size]
            batch_token_load = sum(requests[i].prompt_length for i in batch)

            service_time = service_model.get_service_time(batch_token_load)
            current_time += service_time

            for i in batch:
                requests[i].prefill_completed = True
                prefill_queue.remove(i)
                active_decoding.append(i)
                # TTFT: time from arrival to end of its (batched) prefill
                ttft[i] = current_time - requests[i].arrival_time

            record_event(
                current_time,
                "PREFILL_BATCH",
                batch,
                batch_size=len(batch),
                batch_token_load=batch_token_load,
                service_time=service_time,
            )

        # Priority 2: If no jobs are waiting for prefill, decode one token
        # for each active job in a single batched decode operation.
        elif active_decoding:
            batch = active_decoding[:]
            batch_token_load = len(batch)  # 1 token per active job

            service_time = service_model.get_service_time(batch_token_load)
            current_time += service_time

            newly_completed = []
            for i in batch:
                requests[i].tokens_decoded += 1
                # All jobs in this batch share the same inter-token time, since
                # the GPU advances them together.
                tbt_times[i].append(service_time)

                if requests[i].is_completed:
                    newly_completed.append(i)
                    active_decoding.remove(i)
                    completed.append(i)
                    completion_times[i] = current_time

            record_event(
                current_time,
                "DECODE_BATCH",
                batch,
                batch_size=len(batch),
                batch_token_load=batch_token_load,
                service_time=service_time,
                completed=newly_completed,
            )

        # If the GPU is idle and there is nothing to prefill or decode yet,
        # jump forward in time to the next arrival.
        else:
            if next_arrival_idx < num_jobs:
                current_time = arrival_times[next_arrival_idx]
            else:
                # No more future arrivals and nothing left to process
                break

    # Compute per-job average time between tokens (TBT)
    tbt = np.array([
        np.mean(tbt_times[i]) if tbt_times[i] else 0.0
        for i in range(num_jobs)
    ])

    # For aggregate metrics we only need the overall horizon
    total_time = current_time - arrival_times[0]
    throughput = num_jobs / total_time if total_time > 0 else 0.0

    return {
        "arrival_times": arrival_times,
        "completion_times": completion_times,
        "ttft": ttft,
        "tbt": tbt,
        "throughput": throughput,
        "events": events,
    }


## Validation

# Validate Scheduler A
def validate_mm1(num_jobs=10000):
    lambda_rate = 2.0
    mean_service = 0.4  # 1/μ

    service_model = ServiceTimeModel(
        mean_setup_cost=mean_service,
        mean_marginal_cost=0.0,  # Set to 0 so service = C only
        b0_threshold=0
    )

    results = simulate_run_to_completion(
        num_jobs=num_jobs,
        lambda_rate=lambda_rate,
        prompt_length=1,  # Minimal
        output_budget=0,  # NO DECODE
        service_model=service_model,
        log_events=False
    )

    # Calculate metrics
    response_times = results['completion_times'] - results['arrival_times']
    mean_response = np.mean(response_times)

    # Theoretical values
    mu = 1.0 / mean_service
    rho = lambda_rate / mu
    theoretical_response = 1.0 / (mu - lambda_rate)

    print(f"ρ (utilization): {rho:.3f}")
    print(f"Simulated mean response time: {mean_response:.4f}")
    print(f"Theoretical (M/M/1): {theoretical_response:.4f}")
    print(f"Error: {abs(mean_response - theoretical_response)/theoretical_response * 100:.2f}%")



# Validate Scheduler B

# Light Traffic Validation
def validate_B_light_traffic(
    num_jobs=1000,
    lambda_rate=0.1,
    prompt_length=128,
    output_budget=64,
    mean_setup_cost=0.01,
    mean_marginal_cost=0.001,
    b0_threshold=0,
    max_batch_size=8,
    rng_seed=7,
):
    """
    Light-traffic validation for Scheduler B:
    At very small λ, jobs rarely overlap, so batching is almost never used.
    Throughput should be close to λ and response time close to a single-job service time.
    """
    rng = np.random.default_rng(rng_seed)
    service_model = ServiceTimeModel(
        mean_setup_cost=mean_setup_cost,
        mean_marginal_cost=mean_marginal_cost,
        b0_threshold=b0_threshold,
        rng=rng,
    )

    res = simulate_prefill_priority_batching(
        num_jobs=num_jobs,
        lambda_rate=lambda_rate,
        prompt_length=prompt_length,
        output_budget=output_budget,
        service_model=service_model,
        max_batch_size=max_batch_size,
        rng=rng,
        log_events=False,
    )
    response_times = res["completion_times"] - res["arrival_times"]

    print("Light-traffic validation for Scheduler B")
    print("----------------------------------------")
    print(f"λ = {lambda_rate}")
    print(f"Empirical throughput: {res['throughput']:.4f} jobs/sec (should be ≈ λ)")
    print(f"Mean response time  : {np.mean(response_times):.4f} s")
    print(f"Median response time: {np.median(response_times):.4f} s")
    
    return res, response_times


# Capacity v K validation
def theoretical_service_rate_per_job_closed_form(
    K,
    mean_setup_cost,
    mean_marginal_cost,
    b0,
    tokens_per_job,
):
    """
    Closed-form *approximate* capacity (jobs/sec) for batch size K,
    assuming nearly-full batches of K jobs, each with tokens_per_job tokens,
    and using E[C], E[a] directly from the physics model.
    """
    b = K * tokens_per_job
    excess = max(0, b - b0)
    expected_batch_time = mean_setup_cost + mean_marginal_cost * excess
    if expected_batch_time <= 0:
        return np.nan
    return K / expected_batch_time


def empirical_capacity_from_saturated_run(
    K,
    lambda_rate,
    num_jobs,
    prompt_length,
    output_budget,
    service_model,
    rng,
):
    """
    Estimate an empirical capacity (jobs/sec) for Scheduler B at batch size K
    by running the system under heavy load and using total completed jobs
    divided by total busy time.

    This uses the simulation itself as a 'physics oracle', so it should be
    an upper bound for the achieved throughput in shorter runs.
    """
    res = simulate_prefill_priority_batching(
        num_jobs=num_jobs,
        lambda_rate=lambda_rate,
        prompt_length=prompt_length,
        output_budget=output_budget,
        service_model=service_model,
        max_batch_size=K,
        rng=rng,
        log_events=False,
    )

    # If your results already contain a 'throughput' under heavy load, you
    # can just return that as an empirical upper bound.
    return res["throughput"]


def capacity_vs_K_validation(
    K_values,
    lambda_rate=8.0,
    num_jobs=500,
    warmup=50,
    prompt_length=128,
    output_budget=64,
    mean_setup_cost=0.01,
    mean_marginal_cost=0.001,
    b0_threshold=0,
    replications=25,
    rng_seed=1234,
):
    """
    Validation for Scheduler B.

    For each K:
      - run several replications and report mean +/- std of simulated throughput;
      - compute a closed-form approximate capacity from the S(b) model;
      - optionally, compute an empirical capacity from a long saturated run.

    This makes it clear that:
      * simulated throughput has sampling noise; and
      * the analytic capacity is a rough, usually-conservative approximation.
    """
    rng_master = np.random.default_rng(rng_seed)

    tokens_per_job = prompt_length + output_budget  # rough tokens per job
    sim_mean = []
    sim_std = []
    theo_closed = []
    theo_empirical = []

    for K in K_values:
        throughputs = []

        for _ in range(replications):
            rng = np.random.default_rng(rng_master.integers(1e9))
            service_model = ServiceTimeModel(
                mean_setup_cost=mean_setup_cost,
                mean_marginal_cost=mean_marginal_cost,
                b0_threshold=b0_threshold,
                rng=rng,
            )

            res = simulate_prefill_priority_batching(
                num_jobs=num_jobs,
                lambda_rate=lambda_rate,
                prompt_length=prompt_length,
                output_budget=output_budget,
                service_model=service_model,
                max_batch_size=K,
                rng=rng,
                log_events=False,
            )

            throughputs.append(res["throughput"])

        sim_mean.append(np.mean(throughputs))
        sim_std.append(np.std(throughputs))

        theo_closed.append(
            theoretical_service_rate_per_job_closed_form(
                K=K,
                mean_setup_cost=mean_setup_cost,
                mean_marginal_cost=mean_marginal_cost,
                b0=b0_threshold,
                tokens_per_job=tokens_per_job,
            )
        )

    # Print table with mean +/- std and both capacities
    print("K  sim_mean   sim_std   closed_form_cap  ")
    for K, m, s, tc in zip(K_values, sim_mean, sim_std, theo_closed, ):
        print(f"{K:2d}  {m:8.4f}  {s:8.4f}   {tc:10.4f} ")

    # Plot: simulated mean +/- std vs capacities
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        K_values,
        sim_mean,
        yerr=sim_std,
        fmt="o-",
        label="Simulated throughput (mean ± std)",
    )
    plt.plot(K_values, theo_closed, "s--", label="Closed-form approx capacity")
    plt.xlabel("Max batch size K")
    plt.ylabel("Jobs per second")
    plt.title("Scheduler B: throughput vs approximate and empirical capacity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "K_values": K_values,
        "sim_mean": sim_mean,
        "sim_std": sim_std,
        "closed_form_capacity": theo_closed,
        "empirical_capacity": theo_empirical,
    }


## Experiements:
    
# Batch scaling - Scheduler B
def batching_scale_experiment(
    K_values,
    lambda_rate=1.5,
    num_jobs=500,
    warmup=50,
    prompt_length=128,
    output_budget=64,
    mean_setup_cost=0.01,
    mean_marginal_cost=0.001,
    b0_threshold=0,
    replications=10,
    rng_seed=123,
):
    rng_master = np.random.default_rng(rng_seed)
    results_per_K = []

    for K in K_values:
        throughputs = []
        mean_ttft_list = []
        p95_ttft_list = []
        mean_tbt_list = []
        p95_tbt_list = []

        for _ in range(replications):
            rng = np.random.default_rng(rng_master.integers(1e9))
            service_model = ServiceTimeModel(
                mean_setup_cost=mean_setup_cost,
                mean_marginal_cost=mean_marginal_cost,
                b0_threshold=b0_threshold,
                rng=rng,
            )

            res = simulate_prefill_priority_batching(
                num_jobs=num_jobs,
                lambda_rate=lambda_rate,
                prompt_length=prompt_length,
                output_budget=output_budget,
                service_model=service_model,
                max_batch_size=K,
                rng=rng,
                log_events=False,
            )

            # Discard warm-up jobs
            ttft_full = res["ttft"]
            tbt_full = res["tbt"]

            if warmup >= len(ttft_full):
                continue

            ttft = ttft_full[warmup:]
            tbt = tbt_full[warmup:]

            if len(ttft) == 0 or len(tbt) == 0:
                continue

            throughputs.append(res["throughput"])
            mean_ttft_list.append(np.mean(ttft))
            p95_ttft_list.append(np.percentile(ttft, 95))
            mean_tbt_list.append(np.mean(tbt))
            p95_tbt_list.append(np.percentile(tbt, 95))

        if len(throughputs) == 0:
            print(f"Warning: no valid replications for K={K} (check num_jobs and warmup)")
            continue

        # 95% CI half-width using t-distribution (falls back to 0 if only 1 rep)
        def ci_half_width(samples):
            n = len(samples)
            if n <= 1:
                return 0.0
            mean = np.mean(samples)
            sem = stats.sem(samples)  # standard error
            h = stats.t.ppf(0.975, df=n-1) * sem
            return mean, h

        thr_mean, thr_ci = ci_half_width(throughputs)
        mttft_mean, mttft_ci = ci_half_width(mean_ttft_list)
        p95ttft_mean, p95ttft_ci = ci_half_width(p95_ttft_list)
        mtbt_mean, mtbt_ci = ci_half_width(mean_tbt_list)
        p95tbt_mean, p95tbt_ci = ci_half_width(p95_tbt_list)

        results_per_K.append(
            {
                "K": K,
                "throughput_mean": thr_mean,
                "throughput_ci": thr_ci,
                "mean_TTFT_mean": mttft_mean,
                "mean_TTFT_ci": mttft_ci,
                "p95_TTFT_mean": p95ttft_mean,
                "p95_TTFT_ci": p95ttft_ci,
                "mean_TBT_mean": mtbt_mean,
                "mean_TBT_ci": mtbt_ci,
                "p95_TBT_mean": p95tbt_mean,
                "p95_TBT_ci": p95tbt_ci,
            }
        )

    return results_per_K


# Sensitivity Analysis

# changing c and a

def ca_sensitivity_experiment(
    ca_configs,
    lambdarate=1.5,
    numjobs=1000,
    warmup=100,
    promptlength=128,
    outputbudget=64,
    maxbatchsize=4,
    replications=3,
    rngseed=456,
):
    """
    Sensitivity of Scheduler B to (c, a).

    ca_configs: list of (label, mean_setup_cost, mean_marginal_cost)
    """
    rng_master = np.random.default_rng(rngseed)
    results = []

    for label, mean_setup_cost, mean_marginal_cost in ca_configs:
        thr, mean_ttft, mean_tbt = [], [], []

        for _ in range(replications):
            rng = np.random.default_rng(rng_master.integers(1e9))

            servicemodel = ServiceTimeModel(
                mean_setup_cost,
                mean_marginal_cost,
                0,  # b0_threshold
                rng,
            )

            res = simulate_prefill_priority_batching(
                num_jobs=numjobs,
                lambda_rate=lambdarate,
                prompt_length=promptlength,
                output_budget=outputbudget,
                service_model=servicemodel,
                max_batch_size=maxbatchsize,
                rng=rng,
                log_events=False,
            )

            ttft_full = res["ttft"]
            tbt_full = res["tbt"]

            if warmup < len(ttft_full):
                ttft = ttft_full[warmup:]
                tbt = tbt_full[warmup:]
            else:
                ttft, tbt = ttft_full, tbt_full

            if len(ttft) == 0:
                continue

            thr.append(res["throughput"])
            mean_ttft.append(np.mean(ttft))
            mean_tbt.append(np.mean(tbt))

        if len(thr) == 0:
            print(f"Warning: no valid reps for {label}")
            continue

        results.append(
            {
                "label": label,
                "throughput": np.mean(thr),
                "mean_TTFT": np.mean(mean_ttft),
                "mean_TBT": np.mean(mean_tbt),
            }
        )

    return results

# Long/Short Job Sensitivity

def run_schedulerB_regime(
    label,
    lambdarate,
    numjobs,
    warmup,
    promptlength,
    outputbudget,
    mean_setup_cost=0.01,
    mean_marginal_cost=0.001,
    maxbatchsize=4,
    replications=3,
    rngseed=789,
):
    rng_master = np.random.default_rng(rngseed)
    thr, mean_ttft, p95_ttft = [], [], []

    for _ in range(replications):
        rng = np.random.default_rng(rng_master.integers(1e9))
        servicemodel = ServiceTimeModel(
            mean_setup_cost,
            mean_marginal_cost,
            0,
            rng,
        )
        res = simulate_prefill_priority_batching(
            num_jobs=numjobs,
            lambda_rate=lambdarate,
            prompt_length=promptlength,
            output_budget=outputbudget,
            service_model=servicemodel,
            max_batch_size=maxbatchsize,
            rng=rng,
            log_events=False,
        )

        ttft_full = res["ttft"]
        if warmup < len(ttft_full):
            ttft = ttft_full[warmup:]
        else:
            ttft = ttft_full

        if len(ttft) == 0:
            continue

        thr.append(res["throughput"])
        mean_ttft.append(np.mean(ttft))
        p95_ttft.append(np.percentile(ttft, 95))

    return {
        "label": label,
        "throughput": np.mean(thr),
        "mean_TTFT": np.mean(mean_ttft),
        "p95_TTFT": np.mean(p95_ttft),
    }


# Scheduler A and B comparison

#  Sweep function for Scheduler A vs B with CIs
def run_scheduler_comparison_sweep(
    lambda_values,
    num_jobs=500,
    prompt_length=128,
    output_budget=64,
    service_model=None,
    max_batch_size=4,
    rng_seed=42
):
    # Store results
    results = {
        "lambda": lambda_values,
        "A_ttft_mean": [], "A_ttft_ci": [],
        "A_tbt_mean": [], "A_tbt_ci": [],
        "B_ttft_mean": [], "B_ttft_ci": [],
        "B_tbt_mean": [], "B_tbt_ci": [],
    }

    rng = np.random.default_rng(rng_seed)

    for lam in lambda_values:
        # Scheduler A
        res_a = simulate_run_to_completion(
            num_jobs=num_jobs,
            lambda_rate=lam,
            prompt_length=prompt_length,
            output_budget=output_budget,
            service_model=service_model,
            rng=rng,
            log_events=False
        )

        # Scheduler B
        res_b = simulate_prefill_priority_batching(
            num_jobs=num_jobs,
            lambda_rate=lam,
            prompt_length=prompt_length,
            output_budget=output_budget,
            service_model=service_model,
            max_batch_size=max_batch_size,
            rng=rng,
            log_events=False
        )

        # Calculate Means and 95% CIs
        def get_stats(data):
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            n = len(data)
            margin = 1.96 * (std / np.sqrt(n))
            return mean, margin

        a_ttft_m, a_ttft_c = get_stats(res_a['ttft'])
        a_tbt_m, a_tbt_c = get_stats(res_a['tbt'])
        b_ttft_m, b_ttft_c = get_stats(res_b['ttft'])
        b_tbt_m, b_tbt_c = get_stats(res_b['tbt'])

        results["A_ttft_mean"].append(a_ttft_m)
        results["A_ttft_ci"].append(a_ttft_c)
        results["A_tbt_mean"].append(a_tbt_m)
        results["A_tbt_ci"].append(a_tbt_c)

        results["B_ttft_mean"].append(b_ttft_m)
        results["B_ttft_ci"].append(b_ttft_c)
        results["B_tbt_mean"].append(b_tbt_m)
        results["B_tbt_ci"].append(b_tbt_c)

    return results


## Additional GPUs + Experiements

# --- 1. Define the GPU Worker Class ---
class GPUWorker:
    """
    Represents a single GPU worker in a multi-GPU cluster.
    Tracks its own availability time and processes assigned jobs.
    """
    def __init__(self, worker_id: int, service_model: ServiceTimeModel):
        self.worker_id = worker_id
        self.service_model = service_model
        self.free_time = 0.0  # Time when this GPU becomes idle

    def schedule_job(self, request: Request, current_time: float) -> dict:
        """
        Schedules a job on this worker using Run-to-Completion (Scheduler A) logic.

        Args:
            request: The job request object
            current_time: The time the dispatcher assigns the job

        Returns:
            dict: Event details and metrics for this job
        """
        # The job can start when the GPU is free OR when it arrives, whichever is later
        start_time = max(current_time, self.free_time)

        # --- Prefill Phase ---
        # The prompt is processed as a single batch (Scheduler A logic)
        prefill_duration = self.service_model.get_service_time(request.prompt_length)
        prefill_end = start_time + prefill_duration

        # --- Decode Phase ---
        # Sequential decoding (Scheduler A logic: batch=1 for each token)
        decode_durations = []
        current_decode_time = prefill_end

        for _ in range(request.output_budget):
            step_time = self.service_model.get_service_time(batch_token_load=1)
            current_decode_time += step_time
            decode_durations.append(step_time)

        finish_time = current_decode_time

        # Update worker state: It is now busy until finish_time
        self.free_time = finish_time

        # Compute Metrics
        # TTFT: Time from arrival to end of prefill
        ttft = prefill_end - request.arrival_time
        # TBT: Average time per decoded token
        tbt = np.mean(decode_durations) if decode_durations else 0.0

        return {
            "worker_id": self.worker_id,
            "start_time": start_time,
            "prefill_end": prefill_end,
            "completion_time": finish_time,
            "ttft": ttft,
            "tbt": tbt
        }

# --- 2. Define the Multi-GPU Simulation Loop ---
def simulate_multi_gpu(
    num_gpus: int,
    num_jobs: int,
    lambda_rate: float,
    prompt_length: int,
    output_budget: int,
    service_model: ServiceTimeModel,
    rng=None
):
    """
    Simulates a multi-GPU system with a Least-Load Dispatcher.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Initialize N identical workers
    workers = [GPUWorker(i, service_model) for i in range(num_gpus)]

    # Generate Arrivals (Poisson Process)
    inter_arrivals = rng.exponential(1.0 / lambda_rate, size=num_jobs)
    arrival_times = np.cumsum(inter_arrivals)

    # Storage for results
    ttft_list = np.zeros(num_jobs)
    tbt_list = np.zeros(num_jobs)
    completion_times = np.zeros(num_jobs)

    print(f"Starting Multi-GPU Simulation with {num_gpus} GPUs...")

    for i in range(num_jobs):
        arrival_time = arrival_times[i]
        req = Request(arrival_time, prompt_length, output_budget, request_id=i)

        # --- Dispatcher: Least-Load Strategy ---
        # Find the worker with the minimum 'free_time' (earliest availability)
        target_worker = min(workers, key=lambda w: w.free_time)

        # Schedule the job on that worker
        job_result = target_worker.schedule_job(req, arrival_time)

        # Record metrics
        ttft_list[i] = job_result["ttft"]
        tbt_list[i] = job_result["tbt"]
        completion_times[i] = job_result["completion_time"]

    total_time = np.max(completion_times) - arrival_times[0]
    throughput = num_jobs / total_time

    return {
        "throughput": throughput,
        "ttft": ttft_list,
        "tbt": tbt_list,
        "completion_times": completion_times
    }


# GPU Scaling Experiment

def multigpu_scaling_experiment_with_ci(
    gpu_counts=(1, 2, 4, 8),
    numjobs=2000,
    warmup=300,
    lambdarate=4.0,
    promptlength=128,
    outputbudget=64,
    mean_setup_cost=0.01,
    mean_marginal_cost=0.001,
    b0_threshold=0,
    replications=5,
    rngseed=123,
):
    """
    Sweep number of GPUs and estimate mean throughput, mean TTFT, P95 TTFT
    with 95% confidence intervals across replications.
    """
    rng_master = np.random.default_rng(rngseed)
    all_results = []

    for n_gpus in gpu_counts:
        rep_throughput = []
        rep_mean_ttft = []
        rep_p95_ttft = []

        for _ in range(replications):
            rng = np.random.default_rng(rng_master.integers(1e9))

            servicemodel = ServiceTimeModel(
                mean_setup_cost,
                mean_marginal_cost,
                b0_threshold,
                rng,
            )

            res = simulate_multi_gpu(
                num_gpus=n_gpus,
                num_jobs=numjobs,
                lambda_rate=lambdarate,
                prompt_length=promptlength,
                output_budget=outputbudget,
                service_model=servicemodel,
                rng=rng,
            )

            ttft_full = res["ttft"]
            if warmup < len(ttft_full):
                ttft = ttft_full[warmup:]
            else:
                ttft = ttft_full

            if len(ttft) == 0:
                continue

            rep_throughput.append(res["throughput"])
            rep_mean_ttft.append(np.mean(ttft))
            rep_p95_ttft.append(np.percentile(ttft, 95))

        n = len(rep_throughput)
        if n == 0:
            print(f"Warning: no valid replications for {n_gpus} GPUs")
            continue

        def mean_ci(x):
            x = np.array(x)
            m = x.mean()
            if n > 1:
                se = x.std(ddof=1) / np.sqrt(n)
                ci_half = 1.96 * se
            else:
                ci_half = 0.0
            return m, ci_half

        thr_mean, thr_ci = mean_ci(rep_throughput)
        ttft_mean, ttft_ci = mean_ci(rep_mean_ttft)
        ttft95_mean, ttft95_ci = mean_ci(rep_p95_ttft)

        all_results.append(
            {
                "num_gpus": n_gpus,
                "throughput_mean": thr_mean,
                "throughput_ci": thr_ci,
                "mean_TTFT_mean": ttft_mean,
                "mean_TTFT_ci": ttft_ci,
                "p95_TTFT_mean": ttft95_mean,
                "p95_TTFT_ci": ttft95_ci,
            }
        )

    return all_results










