import cv2
import numpy as np
from PIL import Image
import easyocr
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from typing import List, Dict, Any

class DocumentProcessor:
    def __init__(self):
        # Initialize OCR readers
        self.easyocr_reader = easyocr.Reader(['en'])  # English
        # Initialize BLIP model for visual content analysis
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        # Set device to CPU
        self.device = torch.device("cpu")
        self.blip_model.to(self.device)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return thresh
    
    def extract_text_easyocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text using EasyOCR (better for handwriting)"""
        results = self.easyocr_reader.readtext(image)
        
        extracted_text = []
        for (bbox, text, confidence) in results:
            if confidence > 0.3:  # Filter low confidence results
                extracted_text.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox,
                    'method': 'easyocr'
                })
        
        return extracted_text
    
    def extract_text_tesseract(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text using Tesseract OCR"""
        # Configure Tesseract for English and Malayalam
        config = '--oem 3 --psm 6 -l eng+mal'
        
        # Get detailed data
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        
        extracted_text = []
        n_boxes = len(data['level'])
        
        for i in range(n_boxes):
            confidence = int(data['conf'][i])
            text = data['text'][i].strip()
            
            if confidence > 30 and text:  # Filter low confidence and empty results
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                extracted_text.append({
                    'text': text,
                    'confidence': confidence / 100.0,  # Normalize to 0-1
                    'bbox': [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                    'method': 'tesseract'
                })
        
        return extracted_text
    
    def detect_visual_elements(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect and analyze visual elements like charts and graphs"""
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Use BLIP to generate captions for visual elements
        inputs = self.blip_processor(pil_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True)
        
        # Simple heuristic to detect charts/graphs
        chart_keywords = ['chart', 'graph', 'plot', 'diagram', 'table', 'bar', 'line', 'pie']
        is_visual_element = any(keyword in caption.lower() for keyword in chart_keywords)
        
        visual_elements = []
        if is_visual_element or len(caption.split()) > 3:  # If caption is descriptive
            visual_elements.append({
                'type': 'visual_element',
                'description': caption,
                'confidence': 0.8,  # Default confidence for visual elements
                'bbox': [[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]]
            })
        
        return visual_elements
    
    def process_document_page(self, image: np.ndarray) -> Dict[str, Any]:
        """Process a single document page"""
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Extract text using both methods
        easyocr_results = self.extract_text_easyocr(image)
        tesseract_results = self.extract_text_tesseract(processed_image)
        
        # Detect visual elements
        visual_elements = self.detect_visual_elements(image)
        
        # Combine results
        all_text_results = easyocr_results + tesseract_results
        
        # Sort by confidence and remove duplicates
        all_text_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'text_extractions': all_text_results,
            'visual_elements': visual_elements,
            'page_summary': self._generate_page_summary(all_text_results, visual_elements)
        }
    
    def _generate_page_summary(self, text_results: List[Dict], visual_elements: List[Dict]) -> str:
        """Generate a summary of the page content"""
        texts = [item['text'] for item in text_results if item['confidence'] > 0.5]
        combined_text = ' '.join(texts[:10])  # Take top 10 confident text extractions
        
        visual_descriptions = [elem['description'] for elem in visual_elements]
        visual_summary = ' '.join(visual_descriptions)
        
        summary = f"Text content: {combined_text}"
        if visual_summary:
            summary += f" Visual elements: {visual_summary}"
        
        return summary
    
    def process_multi_page_document(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Process multiple pages of a document"""
        results = []
        for i, image in enumerate(images):
            page_result = self.process_document_page(image)
            page_result['page_number'] = i + 1
            results.append(page_result)
        
        return results
