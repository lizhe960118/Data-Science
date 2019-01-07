def mySqrt(x):
    if x <= 1:
        return x
    left = 0
    right = x
    while(left < right):
        mid = (left + right) // 2
        if (x / mid > mid):
            left = mid + 1
        elif (x < mid * mid):
            right = mid
        else:
            return mid
    return right-1

print(mySqrt(10))