"""
Question:
Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and using same element twice is not allowed.
"""
# Given nums = [2, 7, 11, 15], target = 9
# Because num[0] + num [1] = 2 + 7 = 9
# return [0, 1]

class Solution:
    def twoSum(self, nums: 'List[int]', target: 'int'):
        numMap = {}
        for i in range(len(nums)):
            if numMap.__contains__(target-nums[i]):
                return [numMap.get(target-nums[i]), i]
            else:
                numMap[nums[i]] = i

# Instead of O(n2)
# This is a algorithm with O(n)