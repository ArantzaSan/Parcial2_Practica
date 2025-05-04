from collections import defaultdict

def restrict(factor, variable, value):
    """Restrict a factor to a specific value of a variable."""
    restricted_factor = {}
    for assignment, prob in factor.items():
        if assignment[variable] == value:
            new_assignment = {k: v for k, v in assignment.items() if k != variable}
            restricted_factor[tuple(new_assignment.items())] = prob
    return restricted_factor

def multiply_factors(factor1, factor2):
    """Multiply two factors."""
    result = defaultdict(float)
    for assignment1, prob1 in factor1.items():
        for assignment2, prob2 in factor2.items():
            if all(assignment1.get(var) == assignment2.get(var) for var in set(assignment1) & set(assignment2)):
                new_assignment = {**assignment1, **assignment2}
                result[tuple(new_assignment.items())] += prob1 * prob2
    return result

def sum_out(factor, variable):
    """Sum out a variable from a factor."""
    summed_out_factor = defaultdict(float)
    for assignment, prob in factor.items():
        new_assignment = {k: v for k, v in assignment.items() if k != variable}
        summed_out_factor[tuple(new_assignment.items())] += prob
    return summed_out_factor

def variable_elimination(factors, query_vars, hidden_vars):
    """Perform variable elimination."""
    for var in hidden_vars:
        # Multiply all factors containing the variable
        relevant_factors = [factor for factor in factors if any(var in assignment for assignment in factor)]
        product = relevant_factors[0]
        for factor in relevant_factors[1:]:
            product = multiply_factors(product, factor)
        # Sum out the variable
        summed_out = sum_out(product, var)
        # Remove old factors and add the new one
        factors = [factor for factor in factors if factor not in relevant_factors]
        factors.append(summed_out)
    # Multiply remaining factors
    result = factors[0]
    for factor in factors[1:]:
        result = multiply_factors(result, factor)
    # Normalize the result
    total_prob = sum(result.values())
    normalized_result = {k: v / total_prob for k, v in result.items()}
    return normalized_result

# Example usage
if __name__ == "__main__":
    # Define factors as dictionaries
    factor1 = {('A', True): 0.2, ('A', False): 0.8}
    factor2 = {('A', True, 'B', True): 0.5, ('A', True, 'B', False): 0.5,
               ('A', False, 'B', True): 0.3, ('A', False, 'B', False): 0.7}
    factors = [factor1, factor2]
    query_vars = ['B']
    hidden_vars = ['A']

    result = variable_elimination(factors, query_vars, hidden_vars)
    print("Result:", result)