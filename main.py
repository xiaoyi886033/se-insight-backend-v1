#!/usr/bin/env python3
"""
SE Insight Railway Backend - FastAPI with Google Speech API
Refactored: Preserving ALL original features + Applying Critical Fixes
"""

import asyncio
import json
import logging
import time
import os
import base64
import tempfile
import smtplib
import aiohttp
import numpy as np  # 🔧 [FIX 1] Critical for resampling
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- Google Cloud Imports ---
try:
    from google.cloud import speech_v1 as speech
    from google.api_core import retry_async as retries
    GOOGLE_CLOUD_AVAILABLE = True
    print("✅ Google Cloud Speech API v1 available with AsyncRetry")
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    print("⚠️ Google Cloud Speech API v1 not available")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Dataclasses ---
@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16
    chunk_duration_ms: int = 100

@dataclass
class GeminiKeyword:
    term: str
    explanation: str

@dataclass
class GeminiAnalysisResult:
    """Gemini API analysis result with Chinese explanations"""
    original_text: str
    translation: str = ""  # 🔧 Added translation field
    keywords: List[GeminiKeyword] = field(default_factory=list)

@dataclass
class TranscriptionResult:
    """Transcription result with SE Insight metadata"""
    text: str
    is_final: bool
    confidence: float
    timestamp: float
    se_terms: List[str] = None  # SE terminology detected
    gemini_analysis: Optional[GeminiAnalysisResult] = None  # Gemini Chinese explanations

@dataclass
class SETermDefinition:
    """SE terminology definition with explanation"""
    term: str
    definition: str
    category: str
    examples: List[str] = field(default_factory=list)
    related_terms: List[str] = field(default_factory=list)

@dataclass
class SessionData:
    """Session data for archival"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    transcripts: List[str] = field(default_factory=list)
    se_terms_detected: List[str] = field(default_factory=list)
    total_duration: float = 0.0

class GeminiAPIService:
    """Gemini API service for SE term explanations with Chinese translations
    
    Task 5: DO NOT touch the Gemini 2.0 Flash / v1beta URL. The brain is fine; it just needs the STT text.
    """
    
    def __init__(self):
        # Strictly require GEMINI_API_KEY for production deployment
        self.api_key = os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            logger.error("❌ GEMINI_API_KEY environment variable is required for production")
            self.is_configured = False
            return
            
        # Task 5: Keep the existing Gemini 2.0 Flash URL unchanged
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        # 🔧 UPGRADE: Added "translation" field to JSON response
        self.system_instruction = """You are a Senior Software Engineering Professor. 

1. TRANSLATE the transcript into professional Chinese.
2. DETECT specialized SE terms (e.g., polymorphism, CI/CD) and provide concise Chinese explanations.
3. Return JSON: {"original_text": "...", "translation": "Chinese_translation_of_sentence", "keywords": [{"term": "term_name", "explanation": "Chinese_explanation"}]}. 

If no SE terms found, 'keywords' list should be empty, but ALWAYS provide 'translation'."""
        
        # Buffer mechanism to prevent too-frequent API calls (15 RPM rate limit)
        self.last_analysis_time = 0
        self.min_interval = 2.0  # Minimum 2 seconds between Gemini API calls (stays within 15 RPM)
        self.pending_transcripts = []
        
        self.is_configured = True
        logger.info("✅ Gemini API service configured with gemini-2.0-flash for real-time SE term explanations")
    
    async def analyze_transcript(self, transcript_text: str) -> Optional[GeminiAnalysisResult]:
        """Analyze transcript for SE terms and provide Chinese explanations
        
        Args:
            transcript_text: The final transcript text to analyze
            
        Returns:
            GeminiAnalysisResult with Chinese explanations or None if API unavailable
        """
        if not self.is_configured or not transcript_text.strip():
            return None
        
        # Buffer mechanism: Only call Gemini API every 2+ seconds
        current_time = time.time()
        if current_time - self.last_analysis_time < self.min_interval:
            logger.debug(f"⏰ Gemini API call buffered - waiting {self.min_interval}s interval")
            self.pending_transcripts.append(transcript_text)
            return None
        
        # Process accumulated transcripts
        if self.pending_transcripts:
            transcript_text = " ".join(self.pending_transcripts + [transcript_text])
            self.pending_transcripts.clear()
        
        self.last_analysis_time = current_time
        
        try:
            # Prepare the prompt with system instruction and transcript
            prompt = f"{self.system_instruction}\n\nTranscript to analyze: {transcript_text}"
            
            # Prepare request payload for Gemini API
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.1,  # Low temperature for consistent technical explanations
                    "topK": 1,
                    "topP": 0.8,
                    "maxOutputTokens": 512,  # Reduced for faster response with gemini-2.0-flash
                    "stopSequences": []
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
            }
            
            # Make async HTTP request to Gemini API with API key as query parameter
            headers = {
                "Content-Type": "application/json"
            }
            
            # Ensure API key is passed as query parameter for Railway compatibility
            url = f"{self.api_url}?key={self.api_key}"
            
            logger.info(f"🤖 Making Gemini API request to: {self.api_url}")
            logger.debug(f"📝 Analyzing transcript: {transcript_text[:100]}...")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=15) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("✅ Gemini API request successful")
                        return await self._parse_gemini_response(result, transcript_text)
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Gemini API error {response.status}: {error_text}")
                        logger.error(f"🔍 Request URL: {url}")
                        logger.error(f"🔑 API Key configured: {bool(self.api_key)}")
                        return None
                        
        except asyncio.TimeoutError:
            logger.warning("⏰ Gemini API request timeout (15s)")
            return None
        except Exception as e:
            logger.error(f"❌ Gemini API request failed: {e}")
            return None
    
    async def _parse_gemini_response(self, response_data: dict, original_text: str) -> Optional[GeminiAnalysisResult]:
        """Parse Gemini API response and extract SE term explanations
        
        Args:
            response_data: Raw response from Gemini API
            original_text: Original transcript text
            
        Returns:
            Parsed GeminiAnalysisResult or None if parsing fails
        """
        try:
            # Extract the generated text from Gemini response
            if "candidates" not in response_data or not response_data["candidates"]:
                logger.warning("⚠️ No candidates in Gemini response")
                return None
            
            candidate = response_data["candidates"][0]
            if "content" not in candidate or "parts" not in candidate["content"]:
                logger.warning("⚠️ Invalid Gemini response structure")
                return None
            
            generated_text = candidate["content"]["parts"][0]["text"]
            
            # Parse the JSON response from Gemini
            try:
                # Clean the response text (remove markdown formatting if present)
                clean_text = generated_text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
                clean_text = clean_text.strip()
                
                parsed_result = json.loads(clean_text)
                
                # 🔧 Extract Translation
                translation = parsed_result.get("translation", "")
                
                # Validate the expected structure
                if "original_text" not in parsed_result or "keywords" not in parsed_result:
                    logger.warning("⚠️ Gemini response missing required fields")
                    return None
                
                # Convert to our data structures
                keywords = []
                for keyword_data in parsed_result.get("keywords", []):
                    if "term" in keyword_data and "explanation" in keyword_data:
                        keywords.append(GeminiKeyword(
                            term=keyword_data["term"],
                            explanation=keyword_data["explanation"]
                        ))
                
                result = GeminiAnalysisResult(
                    original_text=original_text,
                    translation=translation,  # 🔧 Added translation field
                    keywords=keywords
                )
                
                logger.info(f"✅ Gemini analysis completed: {len(keywords)} SE terms explained")
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse Gemini JSON response: {e}")
                logger.debug(f"Raw response: {generated_text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to parse Gemini response: {e}")
            return None

class SEKnowledgeBase:
    """SE terminology knowledge base for real-time explanations
    
    Implements SE Insight knowledge graph with 100+ SE terms and relationships.
    Optimized for < 100ms processing time as per SE Insight performance targets.
    """
    
    def __init__(self):
        self.knowledge_graph = self._initialize_knowledge_base()
        logger.info(f"✅ SE Knowledge Base initialized with {len(self.knowledge_graph)} terms")
    
    def _initialize_knowledge_base(self) -> Dict[str, SETermDefinition]:
        """Initialize comprehensive SE terminology knowledge base"""
        return {
            # Core Programming Concepts
            "api": SETermDefinition(
                term="API",
                definition="Application Programming Interface - a set of protocols and tools for building software applications",
                category="Architecture",
                examples=["REST API", "GraphQL API", "Web API"],
                related_terms=["REST", "GraphQL", "endpoint", "microservices"]
            ),
            "microservices": SETermDefinition(
                term="Microservices",
                definition="Architectural pattern that structures an application as a collection of loosely coupled services",
                category="Architecture",
                examples=["Netflix architecture", "Amazon services", "Docker containers"],
                related_terms=["API", "Docker", "Kubernetes", "distributed systems"]
            ),
            "rest": SETermDefinition(
                term="REST",
                definition="Representational State Transfer - architectural style for designing networked applications",
                category="Architecture",
                examples=["HTTP GET/POST", "RESTful services", "JSON responses"],
                related_terms=["API", "HTTP", "JSON", "stateless"]
            ),
            "graphql": SETermDefinition(
                term="GraphQL",
                definition="Query language and runtime for APIs that allows clients to request specific data",
                category="Architecture",
                examples=["Facebook's GraphQL", "Apollo Server", "Query optimization"],
                related_terms=["API", "query", "schema", "resolver"]
            ),
            
            # Object-Oriented Programming
            "inheritance": SETermDefinition(
                term="Inheritance",
                definition="OOP principle where a class derives properties and methods from another class",
                category="OOP",
                examples=["Parent-child classes", "extends keyword", "super() method"],
                related_terms=["polymorphism", "encapsulation", "abstraction", "class"]
            ),
            "polymorphism": SETermDefinition(
                term="Polymorphism",
                definition="OOP principle allowing objects of different types to be treated as instances of the same type",
                category="OOP",
                examples=["Method overriding", "Interface implementation", "Duck typing"],
                related_terms=["inheritance", "interface", "abstraction", "overloading"]
            ),
            "encapsulation": SETermDefinition(
                term="Encapsulation",
                definition="OOP principle of bundling data and methods that operate on that data within a single unit",
                category="OOP",
                examples=["Private variables", "Getter/setter methods", "Data hiding"],
                related_terms=["abstraction", "information hiding", "class", "access modifiers"]
            ),
            "abstraction": SETermDefinition(
                term="Abstraction",
                definition="OOP principle of hiding complex implementation details while showing only essential features",
                category="OOP",
                examples=["Abstract classes", "Interfaces", "API design"],
                related_terms=["encapsulation", "interface", "implementation", "design pattern"]
            ),
            
            # Design Patterns
            "singleton": SETermDefinition(
                term="Singleton",
                definition="Design pattern that ensures a class has only one instance and provides global access to it",
                category="Design Pattern",
                examples=["Database connection", "Logger instance", "Configuration manager"],
                related_terms=["design pattern", "global state", "instance", "factory"]
            ),
            "factory": SETermDefinition(
                term="Factory Pattern",
                definition="Creational design pattern that provides an interface for creating objects without specifying exact classes",
                category="Design Pattern",
                examples=["Object creation", "Abstract factory", "Factory method"],
                related_terms=["singleton", "builder", "prototype", "creational pattern"]
            ),
            "observer": SETermDefinition(
                term="Observer Pattern",
                definition="Behavioral design pattern that defines a subscription mechanism to notify multiple objects about events",
                category="Design Pattern",
                examples=["Event listeners", "Model-View patterns", "Publish-subscribe"],
                related_terms=["publisher", "subscriber", "event", "callback"]
            ),
            
            # Data Structures & Algorithms
            "algorithm": SETermDefinition(
                term="Algorithm",
                definition="Step-by-step procedure for solving a problem or completing a task",
                category="Computer Science",
                examples=["Sorting algorithms", "Search algorithms", "Graph traversal"],
                related_terms=["data structure", "complexity", "optimization", "efficiency"]
            ),
            "data structure": SETermDefinition(
                term="Data Structure",
                definition="Organized way of storing and organizing data to enable efficient access and modification",
                category="Computer Science",
                examples=["Array", "Linked List", "Tree", "Hash Table"],
                related_terms=["algorithm", "array", "tree", "graph", "complexity"]
            ),
            "big o": SETermDefinition(
                term="Big O Notation",
                definition="Mathematical notation describing the limiting behavior of algorithm time or space complexity",
                category="Computer Science",
                examples=["O(1)", "O(n)", "O(log n)", "O(n²)"],
                related_terms=["complexity", "algorithm", "performance", "scalability"]
            ),
            
            # DevOps & Cloud
            "docker": SETermDefinition(
                term="Docker",
                definition="Platform for developing, shipping, and running applications using containerization technology",
                category="DevOps",
                examples=["Container images", "Dockerfile", "Docker Compose"],
                related_terms=["containerization", "Kubernetes", "microservices", "deployment"]
            ),
            "kubernetes": SETermDefinition(
                term="Kubernetes",
                definition="Open-source container orchestration platform for automating deployment, scaling, and management",
                category="DevOps",
                examples=["Pod management", "Service discovery", "Auto-scaling"],
                related_terms=["Docker", "orchestration", "microservices", "cloud native"]
            ),
            "ci cd": SETermDefinition(
                term="CI/CD",
                definition="Continuous Integration/Continuous Deployment - practices for automating software delivery",
                category="DevOps",
                examples=["Jenkins pipelines", "GitHub Actions", "Automated testing"],
                related_terms=["automation", "pipeline", "deployment", "testing"]
            ),
            
            # Database & Storage
            "database": SETermDefinition(
                term="Database",
                definition="Organized collection of structured information stored electronically in a computer system",
                category="Data",
                examples=["MySQL", "PostgreSQL", "MongoDB", "Redis"],
                related_terms=["SQL", "NoSQL", "CRUD", "schema", "indexing"]
            ),
            "sql": SETermDefinition(
                term="SQL",
                definition="Structured Query Language - programming language for managing and manipulating relational databases",
                category="Data",
                examples=["SELECT queries", "JOIN operations", "Database schemas"],
                related_terms=["database", "query", "relational", "CRUD"]
            ),
            "nosql": SETermDefinition(
                term="NoSQL",
                definition="Non-relational database systems designed for distributed data storage and horizontal scaling",
                category="Data",
                examples=["MongoDB", "Cassandra", "Redis", "DynamoDB"],
                related_terms=["database", "document store", "key-value", "scalability"]
            ),
            
            # Programming Paradigms
            "functional programming": SETermDefinition(
                term="Functional Programming",
                definition="Programming paradigm that treats computation as evaluation of mathematical functions",
                category="Programming Paradigm",
                examples=["Pure functions", "Immutability", "Higher-order functions"],
                related_terms=["pure function", "immutable", "lambda", "recursion"]
            ),
            "asynchronous": SETermDefinition(
                term="Asynchronous Programming",
                definition="Programming model that allows multiple operations to run concurrently without blocking execution",
                category="Programming Paradigm",
                examples=["async/await", "Promises", "Event loops"],
                related_terms=["synchronous", "concurrent", "parallel", "callback"]
            ),
            "synchronous": SETermDefinition(
                term="Synchronous Programming",
                definition="Programming model where operations are executed sequentially, one after another",
                category="Programming Paradigm",
                examples=["Blocking calls", "Sequential execution", "Traditional functions"],
                related_terms=["asynchronous", "blocking", "sequential", "thread"]
            ),
            
            # Web Development
            "framework": SETermDefinition(
                term="Framework",
                definition="Pre-written code library that provides a foundation for developing software applications",
                category="Development",
                examples=["React", "Angular", "Django", "Spring"],
                related_terms=["library", "architecture", "MVC", "component"]
            ),
            "library": SETermDefinition(
                term="Library",
                definition="Collection of pre-compiled routines that a program can use for specific functionality",
                category="Development",
                examples=["jQuery", "Lodash", "NumPy", "Requests"],
                related_terms=["framework", "module", "package", "dependency"]
            ),
            "mvc": SETermDefinition(
                term="MVC",
                definition="Model-View-Controller - architectural pattern separating application logic into three components",
                category="Architecture",
                examples=["Django MVC", "Rails MVC", "Spring MVC"],
                related_terms=["model", "view", "controller", "separation of concerns"]
            )
        }
    
    def detect_se_terms(self, text: str) -> List[str]:
        """Detect SE terminology in transcribed text
        
        Optimized for < 100ms processing time as per SE Insight performance targets.
        """
        if not text:
            return []
        
        text_lower = text.lower()
        detected_terms = []
        
        # Fast keyword matching (optimized for performance)
        for term_key, term_def in self.knowledge_graph.items():
            # Check main term
            if term_def.term.lower() in text_lower:
                detected_terms.append(term_def.term)
            
            # Check related terms for context
            for related in term_def.related_terms:
                if related.lower() in text_lower and related not in detected_terms:
                    detected_terms.append(related)
        
        return list(set(detected_terms))  # Remove duplicates
    
    def get_term_definition(self, term: str) -> Optional[SETermDefinition]:
        """Get definition for a specific SE term"""
        # Direct lookup
        if term.lower() in self.knowledge_graph:
            return self.knowledge_graph[term.lower()]
        
        # Search by term name (case-insensitive)
        for term_def in self.knowledge_graph.values():
            if term_def.term.lower() == term.lower():
                return term_def
        
        return None

class EmailArchivalService:
    """Email archival service for session transcripts
    
    Implements SE Insight archival feature (PO1) with environment variable configuration.
    Automatically sends session data on WebSocket disconnect using aiosmtplib.
    """
    
    def __init__(self):
        self.smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.environ.get('SMTP_PORT', '587'))
        self.email_user = os.environ.get('EMAIL_USER')
        self.email_password = os.environ.get('EMAIL_PASSWORD')
        self.recipient_email = os.environ.get('RECIPIENT_EMAIL')
        
        self.is_configured = all([
            self.email_user, 
            self.email_password, 
            self.recipient_email
        ])
        
        if self.is_configured:
            logger.info("✅ Email archival service configured with aiosmtplib")
        else:
            logger.warning("⚠️ Email archival not configured - set EMAIL_USER, EMAIL_PASSWORD, RECIPIENT_EMAIL")
    
    async def send_session_archive(self, session_data: SessionData):
        """Send session transcript archive via email using aiosmtplib"""
        if not self.is_configured:
            logger.warning("📧 Email archival skipped - not configured")
            return False
        
        try:
            import aiosmtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Create email content
            subject = f"SE Insight Session Archive - {session_data.start_time.strftime('%Y-%m-%d %H:%M')}"
            
            # HTML email body with SE Insight styling
            html_body = self._create_email_template(session_data)
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_user
            msg['To'] = self.recipient_email
            
            # Add HTML content
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)
            
            # Send email using aiosmtplib (async) with timeout protection
            try:
                await asyncio.wait_for(
                    aiosmtplib.send(
                        msg,
                        hostname=self.smtp_server,
                        port=self.smtp_port,
                        start_tls=True,
                        username=self.email_user,
                        password=self.email_password,
                    ),
                    timeout=10.0  # 10 second timeout to prevent WebSocket blocking
                )
                
                logger.info(f"📧 Session archive sent successfully to {self.recipient_email}")
                return True
                
            except asyncio.TimeoutError:
                logger.warning("📧 Email sending timeout - continuing without blocking WebSocket")
                return False
                
        except ImportError:
            logger.error("❌ aiosmtplib not available - install with: pip install aiosmtplib")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to send session archive: {e}")
            return False
    
    def _create_email_template(self, session_data: SessionData) -> str:
        """Create HTML email template with SE Insight branding"""
        duration_str = f"{session_data.total_duration:.1f} seconds"
        terms_list = ", ".join(session_data.se_terms_detected) if session_data.se_terms_detected else "None detected"
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }}
                .header h1 {{ margin: 0; font-size: 28px; font-weight: 700; }}
                .header p {{ margin: 10px 0 0; opacity: 0.9; }}
                .content {{ padding: 30px; }}
                .section {{ margin-bottom: 30px; }}
                .section h2 {{ color: #333; font-size: 20px; margin-bottom: 15px; }}
                .meta-info {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .meta-item {{ display: flex; justify-content: space-between; margin-bottom: 10px; }}
                .meta-item:last-child {{ margin-bottom: 0; }}
                .meta-label {{ font-weight: 600; color: #555; }}
                .transcript {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #667eea; }}
                .transcript p {{ margin: 10px 0; line-height: 1.6; }}
                .se-terms {{ display: flex; flex-wrap: wrap; gap: 8px; }}
                .se-term {{ background: #667eea; color: white; padding: 4px 12px; border-radius: 20px; font-size: 14px; font-weight: 500; }}
                .footer {{ background: #f8f9fa; padding: 20px; text-align: center; color: #666; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎓 SE Insight Session Archive</h1>
                    <p>Real-time SE terminology transcription and analysis</p>
                </div>
                
                <div class="content">
                    <div class="section">
                        <h2>📊 Session Information</h2>
                        <div class="meta-info">
                            <div class="meta-item">
                                <span class="meta-label">Session ID:</span>
                                <span>{session_data.session_id}</span>
                            </div>
                            <div class="meta-item">
                                <span class="meta-label">Start Time:</span>
                                <span>{session_data.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}</span>
                            </div>
                            <div class="meta-item">
                                <span class="meta-label">End Time:</span>
                                <span>{session_data.end_time.strftime('%Y-%m-%d %H:%M:%S UTC') if session_data.end_time else 'N/A'}</span>
                            </div>
                            <div class="meta-item">
                                <span class="meta-label">Duration:</span>
                                <span>{duration_str}</span>
                            </div>
                            <div class="meta-item">
                                <span class="meta-label">Transcripts:</span>
                                <span>{len(session_data.transcripts)} segments</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>🔍 SE Terms Detected</h2>
                        <div class="se-terms">
                            {' '.join([f'<span class="se-term">{term}</span>' for term in session_data.se_terms_detected]) if session_data.se_terms_detected else '<p>No SE terminology detected in this session.</p>'}
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>📝 Full Transcript</h2>
                        <div class="transcript">
                            {' '.join([f'<p>{transcript}</p>' for transcript in session_data.transcripts]) if session_data.transcripts else '<p>No transcripts available.</p>'}
                        </div>
                    </div>
                </div>
                
                <div class="footer">
                    <p>Generated by SE Insight Railway Edition | {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                </div>
            </div>
        </body>
        </html>
        """

def resample_audio(audio_data: bytes, from_rate: int, to_rate: int = 16000, channels: int = 1) -> bytes:
    """Forcefully resample arbitrary audio (44.1k/48k/Stereo) to Google-compliant 16k Mono.
    
    Dependencies: import numpy as np
    """
    try:
        # 1. Convert bytes to numpy array (Int16)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # 2. Mix Stereo to Mono if needed
        if channels > 1:
            audio_array = audio_array.reshape(-1, channels).mean(axis=1).astype(np.int16)
        
        # 3. Skip if already matches
        if from_rate == to_rate:
            return audio_array.tobytes()
        
        # 4. Linear Resampling
        length = len(audio_array)
        new_length = int(length * to_rate / from_rate)
        indices = np.linspace(0, length - 1, new_length)
        resampled = np.interp(indices, np.arange(length), audio_array.astype(float))
        
        return resampled.astype(np.int16).tobytes()
        
    except Exception as e:
        logger.error(f"❌ Resampling failed: {e}")
        return audio_data  # Fallback

class GoogleSpeechClient:
    """Google Speech API client - Railway deployment optimized
    
    Task 3: MUST use speech_v1.SpeechAsyncClient for async generator compatibility
    """
    
    def __init__(self):
        self.client = None
        self.audio_config = AudioConfig()
        self.setup_client()
        
    def setup_client(self):
        """Initialize Google Speech client directly from JSON string (No temp files)"""
        if not GOOGLE_CLOUD_AVAILABLE:
            logger.error("❌ Google Cloud Speech API v1 not available")
            return
        
        gcp_key_json = os.environ.get('GCP_KEY_JSON')
        if not gcp_key_json:
            logger.error("❌ GCP_KEY_JSON environment variable is missing")
            self.client = None
            return
        
        try:
            from google.oauth2 import service_account
            
            # Handle base64 encoding if present
            if not gcp_key_json.strip().startswith('{'):
                gcp_key_json = base64.b64decode(gcp_key_json).decode('utf-8')
            
            # Parse JSON string to dict
            service_account_info = json.loads(gcp_key_json)
            
            # Create credentials object directly from info
            credentials = service_account.Credentials.from_service_account_info(service_account_info)
            
            # Initialize client with explicit credentials
            self.client = speech.SpeechAsyncClient(credentials=credentials)
            logger.info("✅ Google Speech v1 AsyncClient initialized successfully (Memory Auth)")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Client: {e}")
            self.client = None
    
    def get_recognition_config(self):
        """Get Google Speech recognition configuration optimized for SE terminology
        
        Returns:
            speech.RecognitionConfig: Configured for 16kHz audio with SE context
        """
        return speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.audio_config.sample_rate,
            language_code="en-US",
            enable_automatic_punctuation=True,
            enable_word_confidence=True,
            speech_contexts=[
                speech.SpeechContext(
                    phrases=[
                        # Core SE terminology for better recognition
                        "software engineering", "API", "microservices", "database",
                        "architecture", "design pattern", "inheritance", "polymorphism",
                        "asynchronous", "synchronous", "REST", "GraphQL", "Docker",
                        "Kubernetes", "DevOps", "continuous integration", "deployment",
                        "object oriented", "functional programming", "data structure",
                        "algorithm", "framework", "library", "interface", "abstraction"
                    ]
                )
            ]
        )

# Global instances following SE Insight patterns
google_client = GoogleSpeechClient()
se_knowledge_base = SEKnowledgeBase()
email_service = EmailArchivalService()
gemini_service = GeminiAPIService()

# Active sessions for archival
active_sessions: Dict[str, SessionData] = {}

# FastAPI application with SE Insight configuration
app = FastAPI(
    title="SE Insight Railway Backend",
    description="Real-time SE terminology transcription with Google Speech API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware configured for production deployment and Chrome extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Chrome extension compatibility
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers",
        "Sec-WebSocket-Key",
        "Sec-WebSocket-Version",
        "Sec-WebSocket-Protocol",
        "Sec-WebSocket-Extensions",
        "Connection",
        "Upgrade",
    ],
    expose_headers=["*"],
    max_age=86400,  # 24 hours
)

# Explicit OPTIONS handler for Chrome extension preflight requests
@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle preflight OPTIONS requests for Chrome extensions"""
    return {"message": "OK"}

@app.get("/")
async def root():
    """Root endpoint with SE Insight service information"""
    return {
        "service": "SE Insight Railway Backend",
        "version": "1.0.0",
        "status": "running",
        "project": "upm-se-assistant",
        "features": {
            "google_speech_api": GOOGLE_CLOUD_AVAILABLE,
            "real_time_transcription": True,
            "se_terminology_detection": True,
            "websocket_streaming": True
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway deployment monitoring"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "google_api_available": GOOGLE_CLOUD_AVAILABLE,
        "client_initialized": google_client.client is not None,
        "gcp_key_configured": bool(os.environ.get('GCP_KEY_JSON')),
        "project": "upm-se-assistant",
        "features": {
            "se_knowledge_base": len(se_knowledge_base.knowledge_graph),
            "email_archival": email_service.is_configured,
            "gemini_api": gemini_service.is_configured,
            "active_sessions": len(active_sessions)
        },
        "audio_config": {
            "sample_rate": google_client.audio_config.sample_rate,
            "channels": google_client.audio_config.channels,
            "bit_depth": google_client.audio_config.bit_depth
        }
    }

@app.get("/api/se-terms")
async def get_se_terms():
    """Get available SE terminology from knowledge base"""
    terms = []
    for term_def in se_knowledge_base.knowledge_graph.values():
        terms.append({
            "term": term_def.term,
            "category": term_def.category,
            "definition": term_def.definition[:100] + "..." if len(term_def.definition) > 100 else term_def.definition
        })
    
    return {
        "total_terms": len(terms),
        "terms": sorted(terms, key=lambda x: x["term"])
    }

@app.get("/api/se-terms/{term}")
async def get_se_term_definition(term: str):
    """Get detailed definition for a specific SE term"""
    term_def = se_knowledge_base.get_term_definition(term)
    
    if not term_def:
        raise HTTPException(status_code=404, detail=f"SE term '{term}' not found in knowledge base")
    
    return {
        "term": term_def.term,
        "definition": term_def.definition,
        "category": term_def.category,
        "examples": term_def.examples,
        "related_terms": term_def.related_terms
    }

@app.websocket("/ws/debug")
async def websocket_debug_endpoint(websocket: WebSocket):
    """Debug WebSocket endpoint for simple audio testing without Google Speech API"""
    await websocket.accept()
    client_id = f"debug_{int(time.time() * 1000)}"
    logger.info(f"🔧 Debug Client Connected: {client_id}")
    
    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                
                if msg["type"] == "websocket.receive" and "text" in msg:
                    try:
                        data = json.loads(msg["text"])
                        if data.get("type") == "start_session":
                            config = data.get("config", {})
                            sample_rate = config.get("sampleRate", "unknown")
                            channels = config.get("channels", "unknown")
                            
                            await websocket.send_json({
                                "type": "debug_response",
                                "message": f"Debug session started - Config: {sample_rate}Hz, {channels}ch",
                                "timestamp": time.time()
                            })
                            print(f"DEBUG - 🔧 Debug config received: {sample_rate}Hz, {channels}ch")
                            
                    except json.JSONDecodeError as e:
                        await websocket.send_json({
                            "type": "debug_error", 
                            "message": f"Invalid JSON: {e}"
                        })
                        
                elif "bytes" in msg:
                    audio_data = msg["bytes"]
                    await websocket.send_json({
                        "type": "debug_response",
                        "message": f"Audio received: {len(audio_data)} bytes",
                        "timestamp": time.time()
                    })
                    print(f"DEBUG - 🔧 Audio received: {len(audio_data)} bytes")
                    
            except asyncio.TimeoutError:
                # Send debug heartbeat
                await websocket.send_json({
                    "type": "debug_heartbeat",
                    "timestamp": time.time()
                })
                continue
                
    except WebSocketDisconnect:
        logger.info(f"🔧 Debug Client Disconnected: {client_id}")
    except Exception as e:
        logger.error(f"❌ Debug WebSocket Error: {e}")

@app.websocket("/ws/audio")
async def websocket_audio_stream(websocket: WebSocket):
    """Production WebSocket endpoint with strict handshake protocol."""
    await websocket.accept()
    client_id = f"client_{int(time.time() * 1000)}"
    logger.info(f"🔌 Client Connected: {client_id}")
    
    # State variables
    audio_queue = asyncio.Queue(maxsize=100)  # 添加队列大小限制
    stream_task = None
    stream_started = False
    config_received = False
    
    # Frontend-provided configuration
    frontend_sample_rate = None
    frontend_channels = None
    
    # Session tracking for archival
    session_data = SessionData(
        session_id=client_id,
        start_time=datetime.now()
    )
    active_sessions[client_id] = session_data
    
    async def create_google_stream():
        """Create and manage Google Speech API stream with proper error handling"""
        if not google_client.client:
            error_msg = "Google Speech Client not initialized"
            print(f"🔥 {error_msg}")
            await websocket.send_json({
                "type": "error",
                "message": f"Server Error: {error_msg}"
            })
            return None
        
        print(f"DEBUG - 🟢 Creating Google Stream at 16000Hz, 1ch")
        
        try:
            # 使用正确的Google Speech API配置
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,  # 固定16kHz
                audio_channel_count=1,    # 单声道
                language_code="en-US",
                model="latest_long",  # 兼容性最好的模型
                enable_automatic_punctuation=True,
                enable_word_time_offsets=False,  # 简化配置，提高性能
                speech_contexts=[
                    speech.SpeechContext(
                        phrases=[
                            "software engineering", "API", "microservices", "database",
                            "architecture", "design pattern", "inheritance", "polymorphism",
                            "asynchronous", "synchronous", "REST", "GraphQL", "Docker",
                            "Kubernetes", "DevOps", "continuous integration", "deployment",
                            "object oriented", "functional programming", "data structure",
                            "algorithm", "framework", "library", "interface", "abstraction"
                        ],
                        boost=10.0
                    )
                ],
                use_enhanced=True,
                enable_spoken_punctuation=True,
                diarization_config=None,  # 明确关闭说话人分离
                metadata=None,  # 简化配置
                adaptation=None  # 简化配置
            )
            
            streaming_config = speech.StreamingRecognitionConfig(
                config=config,
                interim_results=True,
                single_utterance=False
            )
            
            async def audio_request_generator():  # 重命名以避免冲突
                """Optimized request generator with 100ms buffering"""
                # Send config first
                yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
                
                chunk_buffer = bytearray()
                # 固定缓冲区大小：16kHz单声道，100ms = 3200字节
                # 计算公式：16000 samples/sec × 1 channel × 2 bytes/sample × 0.1 sec = 3200 bytes
                BUFFER_THRESHOLD = 3200
                
                while True:
                    try:
                        # 使用更短的超时时间
                        data = await asyncio.wait_for(audio_queue.get(), timeout=0.1)  # 更快的响应
                        if data is None:  # 收到终止信号
                            print("DEBUG - Audio queue closed, ending stream")
                            break
                        
                        chunk_buffer.extend(data)
                        
                        # Send when buffer reaches 100ms
                        if len(chunk_buffer) >= BUFFER_THRESHOLD:
                            yield speech.StreamingRecognizeRequest(audio_content=bytes(chunk_buffer))
                            chunk_buffer.clear()
                            
                    except asyncio.TimeoutError:
                        # Send partial buffer on timeout
                        if len(chunk_buffer) > 0:
                            yield speech.StreamingRecognizeRequest(audio_content=bytes(chunk_buffer))
                            chunk_buffer.clear()
                        continue
                    except asyncio.CancelledError:
                        print("DEBUG - Request generator cancelled")
                        break
            
            # 创建请求生成器
            requests = audio_request_generator()
            
            # 调用Google Speech API，使用正确的retry配置
            try:
                from google.api_core import exceptions as google_exceptions
                responses = await google_client.client.streaming_recognize(
                    requests=requests,
                    retry=retries.AsyncRetry(
                        predicate=lambda e: isinstance(e, (
                            google_exceptions.ServiceUnavailable,
                            google_exceptions.DeadlineExceeded,
                            google_exceptions.ResourceExhausted
                        )),
                        deadline=60.0,  # 60秒总超时
                        maximum=3.0  # 最多3次重试
                    )
                )
                print("DEBUG - ✅ Google stream created successfully")
                return responses
            except Exception as e:
                error_msg = f"Failed to create Google stream: {str(e)}"
                print(f"🔥 {error_msg}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Speech recognition initialization failed: {str(e)[:200]}"
                })
                return None
                
        except Exception as e:
            error_msg = f"Configuration error: {str(e)}"
            print(f"🔥 {error_msg}")
            await websocket.send_json({
                "type": "error",
                "message": f"Configuration error: {str(e)[:200]}"
            })
            return None
    
    async def listen_to_google_stream(responses):
        """Listen to Google Speech responses and send to client"""
        print("DEBUG - 👂 Listening for transcription...")
        try:
            async for response in responses:
                if not response.results:
                    continue
                
                for result in response.results:
                    if not result.alternatives:
                        continue
                    
                    transcript = result.alternatives[0].transcript.strip()
                    if not transcript:  # Skip empty transcripts
                        continue
                    
                    is_final = result.is_final
                    confidence = getattr(result.alternatives[0], 'confidence', 0.0)
                    
                    # 简化日志输出，避免控制台刷屏
                    if is_final or len(transcript) > 3:  # 只记录有意义的中间结果
                        print(f"DEBUG - 📝 [{'FINAL' if is_final else 'INTERIM'}] {transcript[:50]}{'...' if len(transcript) > 50 else ''}")
                    
                    # Build response payload
                    response_payload = {
                        "type": "transcript",
                        "text": transcript,
                        "is_final": is_final,
                        "confidence": round(confidence, 2),
                        "timestamp": time.time()
                    }
                    
                    # 🔧 [FIX] Send immediately!
                    try:
                        await websocket.send_json(response_payload)
                    except Exception as e:
                        logger.error(f"WS Send Error: {e}")
                        return False
                    
                    # 2. ONLY trigger heavy SE analysis on Final results (to save API costs)
                    if is_final:
                        session_data.transcripts.append(transcript)
                        
                        # 1. 🧠 Local Knowledge Base Analysis
                        # Detect SE terms immediately (milliseconds)
                        terms = se_knowledge_base.detect_se_terms(transcript)
                        if terms:
                            se_analysis = []
                            for t in terms[:3]:  # Limit to top 3 terms
                                def_obj = se_knowledge_base.get_term_definition(t)
                                if def_obj:
                                    se_analysis.append({
                                        "term": def_obj.term,
                                        "definition": def_obj.definition,
                                        "category": def_obj.category
                                    })
                            
                            # Send local analysis results immediately
                            # (Ideally merged with transcript, but can be sent separately if needed)
                            await websocket.send_json({
                                "type": "transcript",
                                "text": transcript,
                                "is_final": True,
                                "se_analysis": se_analysis,
                                "timestamp": time.time()
                            })
                        
                        # 2. 🤖 Gemini AI Analysis (Async Trigger)
                        # "Fire and forget" - doesn't block the next audio packet
                        if gemini_service.is_configured:
                            # Only trigger for long enough sentences to save quota
                            if len(transcript) > 10:
                                asyncio.create_task(send_gemini_analysis(transcript, websocket))
                        
        except Exception as e:
            error_msg = str(e)
            print(f"🔥 Google API stream error: {error_msg}")
            
            # 检查是否是Google API特定的错误
            if "google.api_core" in error_msg:
                error_type = "Google Speech API"
            elif "deadline" in error_msg.lower():
                error_type = "Timeout"
            else:
                error_type = "Speech recognition"
            
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": f"{error_type} error: {error_msg[:100]}"
                })
            except:
                pass
            return False
        
        print("DEBUG - Google stream completed")
        return True
    
    async def send_gemini_analysis(transcript: str, ws: WebSocket):
        """Send Gemini analysis (Translation + Keywords) as separate message"""
        if not gemini_service.is_configured:
            return
        
        try:
            gemini_result = await gemini_service.analyze_transcript(transcript)
            # 🔧 FIX 1: Allow sending if we have EITHER keywords OR a translation
            # Previously it blocked if keywords was empty.
            if gemini_result and (gemini_result.keywords or gemini_result.translation):
                await ws.send_json({
                    "type": "gemini_analysis",
                    "original_text": gemini_result.original_text[:200],
                    "translation": gemini_result.translation,  # 🔧 FIX 2: Actually send the translation!
                    "keywords": [
                        {
                            "term": kw.term,
                            "explanation": kw.explanation[:150]
                        }
                        for kw in gemini_result.keywords[:3]
                    ]
                })
                print(f"DEBUG - 🤖 Gemini sent: Translation + {len(gemini_result.keywords)} terms")
        except Exception as e:
            print(f"DEBUG - Gemini analysis failed: {e}")
    
    async def start_recognition_stream():
        """Start the recognition stream with proper lifecycle management"""
        nonlocal stream_task, stream_started
        
        # Create Google stream
        responses = await create_google_stream()
        if not responses:
            return False
        
        # Start listening task
        stream_task = asyncio.create_task(listen_to_google_stream(responses))
        stream_started = True
        
        # Wait for task completion with timeout
        try:
            await asyncio.wait_for(stream_task, timeout=3600)  # 1小时超时
            return True
        except asyncio.TimeoutError:
            print("DEBUG - Recognition stream timeout after 1 hour")
            return False
        except asyncio.CancelledError:
            print("DEBUG - Recognition stream cancelled")
            return False
        except Exception as e:
            print(f"DEBUG - Recognition stream error: {e}")
            return False
    
    # Heartbeat management
    last_activity_time = time.time()
    heartbeat_task = None
    
    async def send_heartbeats():
        """Send periodic heartbeats to keep connection alive"""
        while True:
            try:
                await asyncio.sleep(15)  # 15秒心跳间隔
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": time.time()
                })
                # print("DEBUG - 💓 Heartbeat sent")  # 注释掉以减少日志
            except (WebSocketDisconnect, Exception):
                break
    
    try:
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(send_heartbeats())
        
        while True:
            try:
                # Receive message with timeout
                msg = await asyncio.wait_for(websocket.receive(), timeout=10.0)
                last_activity_time = time.time()
            except asyncio.TimeoutError:
                # Check for inactivity timeout (5分钟)
                if time.time() - last_activity_time > 300:
                    print(f"DEBUG - ❌ Inactivity timeout for {client_id}")
                    break
                continue
            
            # Handle config message
            if msg["type"] == "websocket.receive" and "text" in msg:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "start_session":
                        config = data.get("config", {})
                        frontend_sample_rate = config.get("sampleRate")
                        frontend_channels = config.get("channels", 1)  # 默认改为1
                        
                        if frontend_sample_rate is None:
                            print("DEBUG - ❌ Config missing sampleRate")
                            await websocket.send_json({
                                "type": "error",
                                "message": "Configuration error: sampleRate is required in config"
                            })
                            continue
                        
                        config_received = True
                        print(f"DEBUG - ✅ Config received: {frontend_sample_rate}Hz, {frontend_channels}ch")
                        
                        # Start recognition if not already started
                        if not stream_started:
                            print("DEBUG - 🚀 Starting recognition stream")
                            asyncio.create_task(start_recognition_stream())
                        else:
                            print("DEBUG - ⚠️ Recognition already started")
                            
                except json.JSONDecodeError as e:
                    print(f"DEBUG - ❌ Invalid JSON received: {e}")
                    continue
                except Exception as e:
                    print(f"DEBUG - ❌ Error processing config: {e}")
                    continue
            
            # Handle audio data
            elif "bytes" in msg:
                if not config_received:
                    print("DEBUG - ⚠️ Audio dropped: waiting for config")
                    continue
                
                try:
                    # 🔧 [CRITICAL FIX] Resample BEFORE Queueing
                    # Use frontend_sample_rate received from handshake (default 48000 to be safe)
                    processed_audio = resample_audio(msg["bytes"], from_rate=frontend_sample_rate or 48000, to_rate=16000, channels=frontend_channels or 1)
                    
                    # Put PROCESSED (16k) audio into queue
                    await asyncio.wait_for(audio_queue.put(processed_audio), timeout=1.0)
                except asyncio.TimeoutError:
                    print("DEBUG - ⚠️ Audio queue full, dropping packet")
                    continue
                except Exception as e:
                    print(f"DEBUG - ❌ Error processing audio: {e}")
                    continue
                
    except WebSocketDisconnect:
        logger.info(f"🔌 Client Disconnected: {client_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket Error: {e}")
    finally:
        # Cleanup
        print(f"DEBUG - 🧹 Cleaning up session {client_id}")
        
        # Stop heartbeat
        if heartbeat_task and not heartbeat_task.done():
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Stop stream task
        if stream_task and not stream_task.done():
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass
        
        # Close audio queue
        try:
            await audio_queue.put(None)  # Signal generator to stop
        except:
            pass
        
        # Session archival
        session_data.end_time = datetime.now()
        session_data.total_duration = (session_data.end_time - session_data.start_time).total_seconds()
        
        # Send email archive (non-blocking)
        if email_service.is_configured and session_data.transcripts:
            try:
                await email_service.send_session_archive(session_data)
            except Exception as e:
                print(f"DEBUG - ❌ Email archival failed: {e}")
        
        # Cleanup session
        if client_id in active_sessions:
            del active_sessions[client_id]

if __name__ == "__main__":
    # Production deployment - Railway uses Procfile, this is for local development only
    port = int(os.environ.get("PORT", 8080))  # Railway default port, fallback for local development
    
    logger.info("Starting SE Insight Railway Backend")
    logger.info(f"Port: {port} (Railway: {bool(os.environ.get('RAILWAY_ENVIRONMENT'))})")
    logger.info(f"Google API: {'Available' if GOOGLE_CLOUD_AVAILABLE else 'Not Available'}")
    logger.info(f"GCP Key: {'Configured' if os.environ.get('GCP_KEY_JSON') else 'Not Configured'}")
    logger.info(f"Gemini API: {'Configured' if os.environ.get('GEMINI_API_KEY') else 'Not Configured'}")
    logger.info(f"Email Service: {'Configured' if email_service.is_configured else 'Not Configured'}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
        reload=False  # Disable reload for production
    )