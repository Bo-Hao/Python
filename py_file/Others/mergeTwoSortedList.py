"""
Merge two sorted linked lists and return it as a new list. The new list should be
made by splicing together the nodes of the first two lists.
"""
# Input 1 -> 2 -> 4, 1 -> 3 -> 4
# Output 1 -> 1 -> 2 -> 3 -> 4 -> 4



# Definition for singly-linked list
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def mergeTwoList(self, l1: 'ListNode', l2: 'ListNode'):
        dum = ListNode(None)
        prev = dum
        
        while l1 and l2:
            if l1.val <= l2.val:
                prev.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next
            prev = prev.next
            
        if l1 == None:
            prev.next = l2
        elif l2 == None:
            prev.next = l1
        
        return dum.next
    

        
