def find_partition(list_test, start, end):
    pivot = list_test[end]
    i = start - 1
    for j in range(start, end):
        if list_test[j] <= pivot:
            i = i + 1
            list_test[i], list_test[j] = list_test[j], list_test[i]
            # 使用一次遍历，将小于pivot的数据移到左边，将大于pivot的数据移到右边
    list_test[i + 1], list_test[end] = list_test[end], list_test[i + 1]
    # 最后将pivot移到对应位置
    return i + 1


def quick_sort(list_test, start, end):
    if start < end:
        middle = find_partition(list_test, start, end)
        quick_sort(list_test, start, middle - 1)
        quick_sort(list_test, middle + 1, end)

if __name__ == '__main__':
    # list_test = [2, 1, 3, 8, 7, 5, 6, 4]
    list_test = [2, 5, 1, 6, 8, 7, 3, 4]
    quick_sort(list_test, 0, len(list_test) - 1)
    print(list_test)


# def quick_sort(arr, frist_index, last_index):
#     if frist_index < last_index:
#         q = parition(arr, frist_index, last_index)
#         quick_sort(arr, frist_index, q)
#         quick_sort(arr, q, last_index)
#      else:
#          return

# def parition(self, arr, frist_index, last_index):
#     i = frist_index - 1
#     for j in range(frist_index, last_index):
#         if arr[j] < :
#             i = i+1
#             arr[i], arr[j] = arr[j], arr[i]
#    arr[i+1], arr[last_index] = arr[last_index] , arr[i+1]
#    return i
 
# quick_sort(arr, 0, len(arr))