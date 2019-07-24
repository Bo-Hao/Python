"""
Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".
"""

# Input: ['flower', 'flow', 'flight']
# return 'fl'

# All given inputs are in lowercase letters a-z 

class Solution:
    def longestCommonPrefix(self, strs: 'List[str]'):
        if not strs: 
            return ""
        s1 = min(strs)
        s2 = max(strs)
        for i , c in enumerate(s1):
            print(i, c)
            if c != s2[i]:
                return s1[:i]
                
        return s1

S = Solution()
print(S.longestCommonPrefix(strs = ['dog', 'flower', 'flow', 'flight']))