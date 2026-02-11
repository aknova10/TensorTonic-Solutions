import numpy as np
def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    #Calculate Po
    np_rater1 = np.array(rater1)
    np_rater2 = np.array(rater2)
    num_agreements = np.sum(np_rater1 == np_rater2)
    Po = num_agreements / len(rater1) #could be len(rater2) as well since length should be the same for either rater to be comparable.


    #Calculate Pe
    total_unique_elements = set(rater1+rater2) #append list and convert to set to get unique values; considers scenario where there is a label in one rater not present in the other
    Pe = 0
    for element in total_unique_elements:
        Pe += (rater1.count(element) / len(rater1)) * (rater2.count(element) / len(rater2))
        
    if Pe == 1:
        return 1

    kappa = (Po - Pe)/(1 - Pe)
    return kappa 