import random
import unittest


def quick_sort(unsorted_list):
    l = len(unsorted_list)
    if l <= 1:
        return unsorted_list
    pivot = unsorted_list[l // 2]
    left = [x for x in unsorted_list if x < pivot]
    middle = [x for x in unsorted_list if x == pivot]
    right = [x for x in unsorted_list if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


def bubble_sort(alist):
    for passnum in range(len(alist) - 1, 0, -1):
        # print(passnum)
        for i in range(passnum):
            if alist[i] > alist[i + 1]:
                alist[i], alist[i + 1] = alist[i + 1], alist[i]
                # print(alist)
    return alist


def merge_sort_python_way(alist):
    if len(alist) > 1:
        mid = len(alist) // 2
        lefthalf = alist[:mid]
        righthalf = alist[mid:]

        merge_sort_python_way(lefthalf)
        merge_sort_python_way(righthalf)

        i = 0
        j = 0
        k = 0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                alist[k] = lefthalf[i]
                i = i + 1
            else:
                alist[k] = righthalf[j]
                j = j + 1
            k = k + 1

        while i < len(lefthalf):
            alist[k] = lefthalf[i]
            i = i + 1
            k = k + 1

        while j < len(righthalf):
            alist[k] = righthalf[j]
            j = j + 1
            k = k + 1
    return alist


def merge_sort(unsorted_list):
    merge_sort_part(unsorted_list, 0, len(unsorted_list) - 1)
    return unsorted_list


def merge_sort_part(unsorted_list, left, right):
    if left < right:
        middle = (left + right) // 2
        merge_sort_part(unsorted_list, left, middle)
        merge_sort_part(unsorted_list, middle + 1, right)
        merge(unsorted_list, left, middle, right)


def merge(unsorted_list, left, middle, right):
    merged_list = []
    saveLeft = left
    p = middle + 1
    while left <= middle and p <= right:
        if unsorted_list[left] < unsorted_list[p]:
            merged_list += [unsorted_list[left]]
            left += 1
        else:
            merged_list += [unsorted_list[p]]
            p += 1
    while left <= middle:
        merged_list.append(unsorted_list[left])
        left += 1
    while p <= right:
        merged_list.append(unsorted_list[p])
        p += 1
    k = 0
    while (k < len(merged_list)):
        unsorted_list[saveLeft + k] = merged_list[k]
        k += 1


def insert_sort(unsorted_list):
    l = len(unsorted_list)
    n = 1
    while n < l:
        j = n
        while j >= 1 and unsorted_list[j] < unsorted_list[j - 1]:
            temp = unsorted_list[j - 1]
            unsorted_list[j - 1] = unsorted_list[j]
            unsorted_list[j] = temp
            j = j - 1
        n = n + 1

    return unsorted_list


class TestStringMethods(unittest.TestCase):
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    def test_quick_sort(self):
        print("test_quick_sort")
        expected_list = list(range(9))
        unsorted_list = random.sample(range(0, 9), 9)
        print(unsorted_list)
        sorted_list = quick_sort(unsorted_list)
        print(sorted_list)
        self.assertEqual(sorted_list, expected_list)

    def test_insert_sort(self):
        print("test_insert_sort")
        expected_list = list(range(9))
        unsorted_list = list(range(9))
        random.shuffle(unsorted_list)
        print(unsorted_list)
        sorted_list = insert_sort(unsorted_list)
        print(sorted_list)
        self.assertEqual(sorted_list, expected_list, "fail sorting")

    def test_merge_sort(self):
        print("test_merge_sort")
        expected_list = list(range(9))
        # expected_list = [1, 2 , 3, 4]
        unsorted_list = list(range(9))
        # unsorted_list = [2, 1 , 4 ,3]
        random.shuffle(unsorted_list)
        print(unsorted_list)
        sorted_list = merge_sort(unsorted_list)
        print(sorted_list)
        self.assertEqual(sorted_list, expected_list, "fail sorting")

    def test_merge_sort_python_way(self):
        print("test_merge_sort_python_way")
        expected_list = list(range(9))
        # expected_list = [1, 2 , 3, 4]
        unsorted_list = list(range(9))
        # unsorted_list = [2, 1 , 4 ,3]
        random.shuffle(unsorted_list)
        print(unsorted_list)
        sorted_list = merge_sort_python_way(unsorted_list)
        print(sorted_list)
        self.assertEqual(sorted_list, expected_list, "fail sorting")

    def test_bubble_sort(self):
        print("test_bubble_sort")
        expected_list = list(range(9))
        # expected_list = [1, 2 , 3, 4]
        unsorted_list = list(range(9))
        # unsorted_list = [2, 1 , 4 ,3]
        random.shuffle(unsorted_list)
        print(unsorted_list)
        sorted_list = bubble_sort(unsorted_list)
        print(sorted_list)
        self.assertEqual(sorted_list, expected_list, "fail sorting")


if __name__ == '__main__':
    unittest.main()
