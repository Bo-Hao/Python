"""
Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0?
Find all unique triplets in the array which gives the sum of zero

The solution set must not contain duplicate triplets.
"""
# Given array nums = [-1, 0, 1, 2, -1, -4]
# A solution set is : 
# [
#   [-1, 0, 1], 
#   [-1, -1, 2]
# ]


class Solution:
    def threeSum(self, nums: 'List[int]'):
        if len(nums) < 3:
            return []
        nums.sort()
        res = set()
        for i , v in enumerate(nums[:-2]):
            if i >= 1 and v == nums[i - 1]:
                continue
            d = {}
            for x in nums[i + 1:]:
                if x not in d:
                    d[- v - x] = 1
                else:
                    res.add((v, -v-x, x))
        return list(res)
        
S = Solution().threeSum([-1, 0, 1, 2, -1, -4])
print(S)
