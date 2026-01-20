def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number using dynamic programming.

    Args:
        n: The position in the Fibonacci sequence

    Returns:
        The nth Fibonacci number
    """

    # <CURSOR HERE - autocomplete should suggest implementation>
    # Expected: if n <= 1: return n
    #           dp = [0] * (n + 1)
    #           dp[1] = 1
    #           for i in range(2, n + 1):
    #               dp[i] = dp[i-1] + dp[i-2]
    #           return dp[n]