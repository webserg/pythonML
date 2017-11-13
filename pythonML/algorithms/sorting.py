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


if __name__ == '__main__':
    unittest.main()
