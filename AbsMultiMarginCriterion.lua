local AbsMultiMarginCriterion, parent = torch.class('nn.AbsMultiMarginCriterion', 'nn.Criterion')

function AbsMultiMarginCriterion:__init(p, weights)
   assert(p == nil or p == 1 or p == 2, 'only p=1 and p=2 supported')
   self.p = p or 1
   parent.__init(self)
   self.sizeAverage = true
   self.weights = weights
   print(self.weights)
end

function AbsMultiMarginCriterion:updateOutput(input, target)
   -- backward compatibility
   self.p = self.p or 1
   return input.nn.AbsMultiMarginCriterion_updateOutput(self, input, target, self.weights)
end

function AbsMultiMarginCriterion:updateGradInput(input, target)
   return input.nn.AbsMultiMarginCriterion_updateGradInput(self, input, target, self.weights)
end
