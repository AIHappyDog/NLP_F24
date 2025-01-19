import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import openai
from typing import Dict, List, Any
import nltk
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
import re
from typing import Dict, List, Any, Optional


class DataCollector:
    def __init__(self):
        self.data = None

    def fetch_data(self, symbol, period="6mo"):
        
        try:
            stock = yf.Ticker(symbol)
            self.data = stock.history(period=period)
            return self.data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

class DataPreprocessor:
    def clean_data(self, data):
        
        if data is None:
            return None

       
        data = data.dropna()

       
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("Missing required columns in data")

        return data

class EMACalculator:
    def calculate_ema(self, data, periods=[10, 30, 50]):
        
        ema_data = pd.DataFrame()

        for period in periods:
            ema_data[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()

        return ema_data

class PatternDetector:
    def detect_patterns(self, price_data, ema_data):
        
        patterns = {
            'crossovers': [],
            'trends': [],
            'current_position': {}
        }

       
        if ema_data['EMA_10'].iloc[-1] > ema_data['EMA_30'].iloc[-1]:
            patterns['crossovers'].append("Short-term EMA above medium-term EMA")
        else:
            patterns['crossovers'].append("Short-term EMA below medium-term EMA")

        # Detect current price position relative to EMAs
        current_price = price_data['Close'].iloc[-1]
        for column in ema_data.columns:
            patterns['current_position'][column] = current_price > ema_data[column].iloc[-1]

        return patterns

class Pipeline:
    def __init__(self, api_key):
        self.api_key = api_key
        self.data_collector = DataCollector()
        self.preprocessor = DataPreprocessor()
        self.ema_calculator = EMACalculator()
        self.pattern_detector = PatternDetector()

    def generate_base_analysis(self, symbol):
        raw_data = self.data_collector.fetch_data(symbol)
        clean_data = self.preprocessor.clean_data(raw_data)

        if clean_data is None:
            return None

        
        ema_data = self.ema_calculator.calculate_ema(clean_data)
        patterns = self.pattern_detector.detect_patterns(clean_data, ema_data)

        
        analysis = {
            'price_data': {
                'current': clean_data['Close'].iloc[-1],
                'previous': clean_data['Close'].iloc[-2],
                'change_percent': ((clean_data['Close'].iloc[-1] / clean_data['Close'].iloc[-2]) - 1) * 100
            },
            'ema_data': {
                col: ema_data[col].iloc[-1] for col in ema_data.columns
            },
            'patterns': patterns
        }

        return analysis

class ModelComparisonPipeline:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key
        self.models = {
            #compare two models
            'gpt-3.5-turbo': {"name": "GPT-3.5", "max_tokens": 4000},
            'gpt-4': {"name": "GPT-4o", "max_tokens": 8000}
        }

    def generate_model_reports(self, analysis_data: Dict) -> Dict:
        reports = {}
        prompt = self._construct_prompt(analysis_data)

        for model_id, model_info in self.models.items():
            try:
                response = openai.ChatCompletion.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are a technical analysis expert explaining market trends to beginners. Focus on objective analysis of EMA indicators without making predictions."},
                        {"role": "user", "content": prompt}
                    ]
                )
                reports[model_info["name"]] = response.choices[0].message['content']
            except Exception as e:
                reports[model_info["name"]] = f"Error generating report: {e}"

        return reports

    def _construct_prompt(self, analysis_data: Dict) -> str:
        template = """
        Generate a technical analysis report based on the following data:

        Current Price: ${current_price:.2f}
        Daily Change: {change_percent:.2f}%

        EMA Analysis:
        {ema_analysis}

        Pattern Analysis:
        {pattern_analysis}

        Requirements:
        1. Provide an objective description of the current trend based on EMA positions
        2. Explain the technical indicators in simple terms
        3. Describe the current market context without making future predictions
        4. Use language suitable for beginner investors
        5. Include clear explanations of any technical terms used
        6. Structure the report in clear sections

        Focus on helping beginners understand what the indicators mean, not what they might predict.
        """

        ema_analysis = "\n".join([
            f"{key}: ${value:.2f}"
            for key, value in analysis_data['ema_data'].items()
        ])

        pattern_analysis = "\n".join([
            f"- {pattern}" for pattern in analysis_data['patterns']['crossovers']
        ])

        return template.format(
            current_price=analysis_data['price_data']['current'],
            change_percent=analysis_data['price_data']['change_percent'],
            ema_analysis=ema_analysis,
            pattern_analysis=pattern_analysis
        )

class ReportEvaluator:
    def __init__(self):
        try:
            nltk.download('punkt', quiet=True)
        except:
            print("NLTK download failed, but continuing...")

    def evaluate_reports(self, reports: Dict[str, str]) -> Dict[str, Dict]:
        evaluations = {}

        for model, report in reports.items():
            evaluations[model] = {
                'objectivity': self._measure_objectivity(report),
                'technical_term_usage': self._analyze_technical_terms(report),
                'structure_quality': self._evaluate_structure(report),
                'explanation_clarity': self._evaluate_explanations(report)
            }

        return evaluations


    def _measure_objectivity(self, text: str) -> float:
        predictive_phrases = ['will', 'should', 'could', 'might', 'predict', 'expect', 'forecast']
        words = text.lower().split()
        predictive_count = sum(1 for word in words if word in predictive_phrases)
        return 1 - (predictive_count / len(words))

    def _analyze_technical_terms(self, text: str) -> Dict:
        technical_terms = ['EMA', 'trend', 'crossover', 'support', 'resistance', 'volume']
        found_terms = {}

        for term in technical_terms:
            term_count = len(re.findall(r'\b' + term + r'\b', text, re.IGNORECASE))
            explanation_exists = 'means' in text.lower().split(term.lower())
            found_terms[term] = {
                'count': term_count,
                'explained': explanation_exists
            }

        return found_terms

    def _evaluate_structure(self, text: str) -> float:
        sections = len(re.findall(r'\n\n', text))
        has_headers = bool(re.search(r'^[A-Z].*:', text, re.MULTILINE))
        return min((sections + int(has_headers)) / 10, 1.0)

    def _evaluate_explanations(self, text: str) -> float:
        explanation_phrases = ['means', 'indicates', 'represents', 'is a', 'refers to']
        explanation_count = sum(text.lower().count(phrase) for phrase in explanation_phrases)
        return min(explanation_count / 10, 1.0)

def run_comparison(api_key: str, symbol: str) -> Dict:
    # Initialize pipeline components
    pipeline = Pipeline(api_key)
    comparison_pipeline = ModelComparisonPipeline(api_key)
    evaluator = ReportEvaluator()

    # Generate base analysis
    analysis = pipeline.generate_base_analysis(symbol)
    if analysis is None:
        return {"error": "Failed to generate base analysis"}

    # Generate reports from different models
    reports = comparison_pipeline.generate_model_reports(analysis)

    # Evaluate reports
    evaluations = evaluator.evaluate_reports(reports)

    return {
        'reports': reports,
        'evaluations': evaluations,
        'analysis_data': analysis
    }

# Example usage
if __name__ == "__main__":
    #API_KEY = ""  
    symbol = "AAPL"

    #results = run_comparison(API_KEY, symbol)

    # Print reports and evaluations
    for model, report in results['reports'].items():
        print(f"\n{model} Report:")
        print("-" * 50)
        print(report)
        print("\nEvaluation:")
        print(results['evaluations'][model])