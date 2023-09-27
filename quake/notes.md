Forecasting_seizure_likelihood_from_cycles_of_reported_events.pdf


* HR Data sampled to 0.2Hz, averaged over 5 min intervals
* Data interpolated as well

Point process?
* [ ] TODO Track down Karoly 2021 paper
    - Karoly, P. J. et al. Cycles in Epilepsy. Nat. Rev. Neurol. (2021).
    - Karoly, P. J. et al. Cycles of self-reported seizure likelihood correspond to yield of diagnostic epilepsy monitoring. Epilepsia 62, 416–425 (2021).
39. Wang, E. T. et al. A Bayesian switching linear dynamical system for estimating seizure chronotypes.
* Range of patient-specific cycle persiods used to asses phase-locking

Using this method, is there a difference between the phases of"big envetsand "small" events?

Data:
* Work in time units of days (days since epoch?)
* Look at point process for a given region across 10 years
    * Candidate cycles?
    * Longest period would be 5 years (1825 days)
    * Maybe take the max period as 70% of this value? 1277.5 days ~ 3.49863 years
    * Nah max value should be 50%, 2.5 years 912 days
    * Min period? 1 month? 1.6438%
* Train on 5 years, evaluate on 5 years


