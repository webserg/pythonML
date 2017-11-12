import random
import unittest


def sort(unsorted_list):
    l = len(unsorted_list)
    if (l <= 1):
        return unsorted_list
    pivot = unsorted_list[l // 2]
    left = [x for x in unsorted_list if x < pivot]
    middle = [x for x in unsorted_list if x == pivot]
    right = [x for x in unsorted_list if x > pivot]
    return sort(left) + middle + sort(right)


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

    def test_sort(self):
        sorterd_list = list(range(9))
        unsorted_list = list(range(9))
        random.shuffle(unsorted_list)
        self.assertEqual(sort(unsorted_list), sorterd_list)


if __name__ == '__main__':
    unittest.main()
