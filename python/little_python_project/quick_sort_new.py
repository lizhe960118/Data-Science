def quicksort(arr):
    return quick_sort(arr, 0, len(arr) - 1)

def quick_sort(arr, start, end):
    if start >= end:
        return
    point = arr[start]
    lp = start
    lr = end
    while lp < lr:
        while arr[lr] > point and lp < lr:
            lr -= 1
        arr[lp] = arr[lr]
        while arr[lp] < point and lp < lr:
            lp += 1
        arr[lr] = arr[lp]
    arr[lp] = point
    quick_sort(arr, start, lp-1)
    quick_sort(arr, lp+1, end)
    return arr

print(quicksort([1,3,2,6,5,4]))