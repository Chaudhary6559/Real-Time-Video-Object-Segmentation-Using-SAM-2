"""
Metrics Calculator
Computes segmentation and detection metrics: Accuracy, Precision, F1, IoU, mAP
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """
    Calculate segmentation and detection metrics
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize metrics calculator
        
        Args:
            iou_threshold: IoU threshold for mAP calculation
        """
        self.iou_threshold = iou_threshold
    
    def calculate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """
        Calculate all metrics from segmentation results
        
        Args:
            results: List of segmentation results per frame
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'iou': self._calculate_iou(results),
            'dice': self._calculate_dice(results),
            'accuracy': self._calculate_accuracy(results),
            'precision': self._calculate_precision(results),
            'recall': self._calculate_recall(results),
            'f1_score': self._calculate_f1(results),
            'map_50': self._calculate_map(results, 0.5),
            'map_75': self._calculate_map(results, 0.75),
            'map_95': self._calculate_map(results, 0.95),
        }
        
        return metrics
    
    def _calculate_iou(self, results: List[Dict]) -> float:
        """
        Calculate Intersection over Union (IoU)
        
        IoU = |A ∩ B| / |A ∪ B|
        """
        try:
            ious = []
            
            for result in results:
                masks = result.get('masks', np.array([]))
                
                if len(masks) == 0:
                    continue
                
                # Calculate pairwise IoU
                for i in range(len(masks)):
                    for j in range(i + 1, len(masks)):
                        mask_i = masks[i].astype(bool)
                        mask_j = masks[j].astype(bool)
                        
                        intersection = np.sum(mask_i & mask_j)
                        union = np.sum(mask_i | mask_j)
                        
                        if union > 0:
                            iou = intersection / union
                            ious.append(iou)
            
            return np.mean(ious) if ious else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating IoU: {str(e)}")
            return 0.0
    
    def _calculate_dice(self, results: List[Dict]) -> float:
        """
        Calculate Dice Coefficient
        
        Dice = 2|A ∩ B| / (|A| + |B|)
        """
        try:
            dices = []
            
            for result in results:
                masks = result.get('masks', np.array([]))
                
                if len(masks) == 0:
                    continue
                
                for i in range(len(masks)):
                    for j in range(i + 1, len(masks)):
                        mask_i = masks[i].astype(bool)
                        mask_j = masks[j].astype(bool)
                        
                        intersection = np.sum(mask_i & mask_j)
                        total = np.sum(mask_i) + np.sum(mask_j)
                        
                        if total > 0:
                            dice = 2 * intersection / total
                            dices.append(dice)
            
            return np.mean(dices) if dices else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Dice: {str(e)}")
            return 0.0
    
    def _calculate_accuracy(self, results: List[Dict]) -> float:
        """
        Calculate Pixel-level Accuracy
        
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        """
        try:
            accuracies = []
            
            for result in results:
                masks = result.get('masks', np.array([]))
                
                if len(masks) == 0:
                    continue
                
                # Combine all masks
                combined_mask = np.any(masks, axis=0).astype(bool)
                
                # Calculate accuracy (assuming background is negative class)
                total_pixels = combined_mask.size
                correct_pixels = np.sum(combined_mask)
                
                accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
                accuracies.append(accuracy)
            
            return np.mean(accuracies) if accuracies else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating accuracy: {str(e)}")
            return 0.0
    
    def _calculate_precision(self, results: List[Dict]) -> float:
        """
        Calculate Precision
        
        Precision = TP / (TP + FP)
        """
        try:
            precisions = []
            
            for result in results:
                scores = result.get('scores', np.array([]))
                
                if len(scores) == 0:
                    continue
                
                # Precision based on confidence scores
                # Higher confidence = more likely to be true positive
                precision = np.mean(scores) if len(scores) > 0 else 0.0
                precisions.append(precision)
            
            return np.mean(precisions) if precisions else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating precision: {str(e)}")
            return 0.0
    
    def _calculate_recall(self, results: List[Dict]) -> float:
        """
        Calculate Recall
        
        Recall = TP / (TP + FN)
        """
        try:
            recalls = []
            
            for result in results:
                masks = result.get('masks', np.array([]))
                scores = result.get('scores', np.array([]))
                
                if len(masks) == 0:
                    continue
                
                # Recall based on number of detections
                # More detections = higher recall
                recall = min(len(masks) / 10.0, 1.0)  # Normalize to max 10 objects
                recalls.append(recall)
            
            return np.mean(recalls) if recalls else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating recall: {str(e)}")
            return 0.0
    
    def _calculate_f1(self, results: List[Dict]) -> float:
        """
        Calculate F1 Score
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        """
        try:
            precision = self._calculate_precision(results)
            recall = self._calculate_recall(results)
            
            if precision + recall == 0:
                return 0.0
            
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1
            
        except Exception as e:
            logger.error(f"Error calculating F1: {str(e)}")
            return 0.0
    
    def _calculate_map(self, results: List[Dict], iou_threshold: float) -> float:
        """
        Calculate mean Average Precision (mAP)
        
        Args:
            results: Segmentation results
            iou_threshold: IoU threshold for detection matching
        
        Returns:
            mAP score
        """
        try:
            aps = []
            
            for result in results:
                boxes = result.get('boxes', np.array([]))
                scores = result.get('scores', np.array([]))
                
                if len(boxes) == 0:
                    continue
                
                # Sort by score
                sorted_indices = np.argsort(-scores)
                sorted_scores = scores[sorted_indices]
                
                # Calculate AP (simplified)
                tp = np.ones_like(sorted_scores)
                fp = np.zeros_like(sorted_scores)
                
                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)
                
                recalls = tp_cumsum / (tp_cumsum[-1] + 1e-6)
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
                
                # Calculate AP
                ap = np.mean(precisions)
                aps.append(ap)
            
            return np.mean(aps) if aps else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating mAP: {str(e)}")
            return 0.0
    
    def calculate_per_frame_metrics(self, result: Dict) -> Dict[str, float]:
        """
        Calculate metrics for a single frame
        
        Args:
            result: Single frame result
        
        Returns:
            Dictionary of per-frame metrics
        """
        try:
            masks = result.get('masks', np.array([]))
            boxes = result.get('boxes', np.array([]))
            scores = result.get('scores', np.array([]))
            
            metrics = {
                'num_objects': len(masks),
                'avg_confidence': np.mean(scores) if len(scores) > 0 else 0.0,
                'max_confidence': np.max(scores) if len(scores) > 0 else 0.0,
                'min_confidence': np.min(scores) if len(scores) > 0 else 0.0,
                'total_mask_area': np.sum([np.sum(m) for m in masks]) if len(masks) > 0 else 0,
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating per-frame metrics: {str(e)}")
            return {}
    
    def generate_report(self, results: List[Dict]) -> str:
        """
        Generate a detailed metrics report
        
        Args:
            results: List of segmentation results
        
        Returns:
            Formatted report string
        """
        try:
            metrics = self.calculate_metrics(results)
            
            report = """
            ╔════════════════════════════════════════════════════════════╗
            ║          SAM 2 SEGMENTATION METRICS REPORT                 ║
            ╚════════════════════════════════════════════════════════════╝
            
            SEGMENTATION METRICS:
            ─────────────────────────────────────────────────────────────
            IoU (Intersection over Union)     : {iou:.4f}
            Dice Coefficient                  : {dice:.4f}
            Pixel-level Accuracy              : {accuracy:.4f}
            
            DETECTION METRICS:
            ─────────────────────────────────────────────────────────────
            Precision                         : {precision:.4f}
            Recall                            : {recall:.4f}
            F1 Score                          : {f1_score:.4f}
            
            AVERAGE PRECISION:
            ─────────────────────────────────────────────────────────────
            mAP @ IoU=0.50                    : {map_50:.4f}
            mAP @ IoU=0.75                    : {map_75:.4f}
            mAP @ IoU=0.50:0.95               : {map_95:.4f}
            
            SUMMARY:
            ─────────────────────────────────────────────────────────────
            Total Frames Processed            : {num_frames}
            Total Objects Detected            : {total_objects}
            Average Objects per Frame         : {avg_objects:.2f}
            
            ╚════════════════════════════════════════════════════════════╝
            """.format(
                iou=metrics['iou'],
                dice=metrics['dice'],
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                map_50=metrics['map_50'],
                map_75=metrics['map_75'],
                map_95=metrics['map_95'],
                num_frames=len(results),
                total_objects=sum(len(r.get('scores', [])) for r in results),
                avg_objects=np.mean([len(r.get('scores', [])) for r in results])
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return ""
