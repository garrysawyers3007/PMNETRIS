import torch
import torch.nn as nn

def L1_loss(pred, target):
  loss = nn.L1Loss()(pred, target)
  return loss

def MSE(pred, target):
  loss = nn.MSELoss()(pred, target)
  return loss

def RMSE(pred, target, metrics=None):
  loss = torch.sqrt(torch.mean(((pred-target)**2)))
  return loss

'''
Building segments loss function: Takes in two np arrays:
When the input is 0, there if the pred is not 0 - then it leads to increase in a count.
This count is averaged acrossed samples.
Need to be between (input,pred) and black pixels in input need to be matched with the black pixels in the pred.
'''

def building_segments(input,pred,target):
  building_prediction = torch.where((input!=0)&(pred!=0),1,0)
  building_input = torch.where((input!=0),1,0)
  building_prediction_sum = building_prediction.sum()
  building_input_sum = building_input.sum()
  avg_across_one_batch = building_prediction_sum/building_input_sum
  '''
  Uncomment the following lines to debug these outputs :
  print("Building_prediction_sum ",building_prediction_sum)
  print("Building input sum ",building_input_sum)
  print("Avg across one batch ",avg_across_one_batch)
  '''
  return avg_across_one_batch

'''
It computes the rmse in the ROI region alone and avoids checking the
building segmentation loss as well that is unrelated to the problem statement.
'''
def roi_rmse_loss(input,pred,target):
  input = input[:,0,:,:].unsqueeze(1)
  #print("Input dimension ",input.shape)
  x = torch.min(input)
  error_tensor = torch.where(input==x,(pred-target)**2,0)
  sum_torch = error_tensor.sum()
  #print("Sum torch ",sum_torch)
  count_non_zero = error_tensor.count_nonzero()
  error_tensor_mean = sum_torch/count_non_zero
  #print("Mean : ",error_tensor_mean)
  #print("Max value: ",error_tensor.max())
  error_float = (error_tensor_mean)**0.5
  #building_sum = torch.sum(building_tensor)
  #print("count_non_zero:",count_non_zero)
  #print("Error float per batch ",error_float)
  return error_float

class WeightedRMSELoss(nn.Module):
  
    def __init__(self, roi_size=32, roi_weight=10.0, non_roi_weight=1.0):
        """
        Args:
            roi_size (int): The height and width of the square RoI box in pixels.
            roi_weight (float): The weight to apply to the loss within the RoI.
            non_roi_weight (float): The weight to apply to the loss outside the RoI.
        """
        super(WeightedRMSELoss, self).__init__()
        self.roi_size = roi_size
        self.roi_weight = roi_weight
        self.non_roi_weight = non_roi_weight
        print(f"WeightedRMSELoss initialized with RoI size: {roi_size}x{roi_size}, RoI weight: {roi_weight}, Non-RoI weight: {non_roi_weight}")

    def forward(self, preds, targets, rx_coords_normalized):
        batch_size, _, height, width = preds.shape
        device = preds.device

        # 1. Map normalized image-space coordinates to pixel coordinates
        # Assumes rx_coords_normalized[0] is x (width) and [1] is y (height)
        # with origin at top-left.
        center_x_px_tensor = rx_coords_normalized[:, 0] * width
        center_y_px_tensor = rx_coords_normalized[:, 1] * height

        # 2. Create the base weight mask
        weight_mask = torch.full_like(preds, self.non_roi_weight, device=device)

        # 3. Populate the RoI in the weight mask for each sample in the batch
        for b in range(batch_size):
            center_x_px = int(center_x_px_tensor[b].item())
            center_y_px = int(center_y_px_tensor[b].item())

            half_roi = self.roi_size // 2
            y1 = max(0, center_y_px - half_roi)
            y2 = min(height, center_y_px + half_roi)
            x1 = max(0, center_x_px - half_roi)
            x2 = min(width, center_x_px + half_roi)
            
            weight_mask[b, :, y1:y2, x1:x2] = self.roi_weight

        # 4. Calculate the weighted squared error and final loss
        squared_error = (preds - targets) ** 2
        weighted_squared_error = squared_error * weight_mask
        
        loss_per_sample = torch.sqrt(torch.mean(weighted_squared_error, dim=[1, 2, 3]))
        final_loss = torch.mean(loss_per_sample)
        
        return final_loss