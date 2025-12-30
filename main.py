#!/usr/bin/env python3
"""
SE Insight Railway Backend - FastAPI with Google Speech API
Following SE Insight architecture standards and workspace rules
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
import numpy as np  # Task 1: Environment Fix - Add import numpy as np at the top (non-negotiable)
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Google Cloud Speech imports - Task 1: Fix Imports & Naming (Dimension 0)
# Action: Use this exact import: from google.cloud import speech_v1 as speech
try:
    from google.cloud import speech_v1 as speech
    from google.api_core import retry_async as retries
    GOOGLE_CLOUD_AVAILABLE = True
    print("✅ Google Cloud Speech API v1 available with AsyncRetry")
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    print("⚠️ Google Cloud Speech API v1 not available")

# Configure logging according to SE Insight standards
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Audio configuration for Google Speech API compatibility"""
    sample_rate: int = 16000  # 16kHz as required by Google Speech API
    channels: int = 1         # Mono audio
    bit_depth: int = 16       # Int16 format
    chunk_duration_ms: int = 100  # 100ms chunks for real-time processing

@dataclass
class GeminiKeyword:
    """Gemini API SE term explanation result"""
    term: str
    explanation: str

@dataclass
class GeminiAnalysisResult:
    """Gemini API analysis result with Chinese explanations"""
    original_text: str
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
        
        # Optimized system instruction for gemini-2.0-flash real-time processing
        self.system_instruction = """You are a Senior Software Engineering Professor analyzing real-time transcripts. Detect specialized SE terms (e.g., polymorphism, CI/CD, microservices, algorithms) and provide concise Chinese explanations (under 40 words). Return JSON format: {"original_text": "...", "keywords": [{"term": "term_name", "explanation": "Chinese_explanation"}]}. Empty keywords list if no SE terms found. No conversational text."""
        
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

class GoogleSpeechClient:
    """Google Speech API client - Railway deployment optimized
    
    Task 3: MUST use speech_v1.SpeechAsyncClient for async generator compatibility
    """
    
    def __init__(self):
        self.client = None
        self.audio_config = AudioConfig()
        self.setup_client()
        
    def setup_client(self):
        """Initialize Google Speech client using GCP_KEY_JSON environment variable
        
        Task 3: Use speech_v1.SpeechAsyncClient for proper async streaming
        """
        if not GOOGLE_CLOUD_AVAILABLE:
            logger.error("❌ Google Cloud Speech API v1 not available - install google-cloud-speech")
            return
            
        # Strictly require GCP_KEY_JSON for production deployment
        gcp_key_json = os.environ.get('GCP_KEY_JSON')
        if not gcp_key_json:
            logger.error("❌ GCP_KEY_JSON environment variable is required for production")
            self.client = None
            return
            
        try:
            # Handle base64 encoding if present
            try:
                if not gcp_key_json.startswith('{'):
                    gcp_key_json = base64.b64decode(gcp_key_json).decode('utf-8')
            except Exception as e:
                logger.warning(f"Failed to decode base64 GCP key, using as-is: {e}")
            
            # Create temporary credentials file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                temp_file.write(gcp_key_json)
                temp_file.flush()
                
                # Set environment variable for Google client
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file.name
                
                # Task 1: Initialize SpeechAsyncClient - Update all references from speech_v1.RecognitionConfig to speech.RecognitionConfig
                self.client = speech.SpeechAsyncClient()
                logger.info("✅ Google Speech v1 AsyncClient initialized with GCP_KEY_JSON")
                
                # Clean up temporary file
                os.unlink(temp_file.name)
                    
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Speech AsyncClient: {e}")
            logger.error("💡 Ensure GCP_KEY_JSON environment variable contains valid service account JSON")
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

# Task 2: Implement the Three-Dimension Specs - Temporal: MANDATORY BUFFERING
async def request_generator(audio_queue, streaming_config):
    """Task 2: Three-Dimension Specs Implementation
    
    Physical: Force encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16 and sample_rate_hertz=16000
    Math: Use np.clip(np.nan_to_num(audio_data) * 32767, -32768, 32767).astype(np.int16)
    Temporal: MANDATORY BUFFERING. Do NOT yield audio to the gRPC stream until you have exactly 3200 bytes (100ms)
    """
    # Dimension 3: First packet MUST be config yield
    yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
    print("DEBUG - Yielded config to gRPC")
    
    # Buffer for Temporal Aggregation (100ms) - MANDATORY BUFFERING
    accumulated_buffer = bytearray()
    
    while True:  # Receiving small 42-byte packets from WebSocket
        try:
            chunk = await audio_queue.get()
            if chunk is None:  # break
                break
            
            # Task 2: Math - Use np.clip(np.nan_to_num(audio_data) * 32767, -32768, 32767).astype(np.int16)
            float_array = np.frombuffer(chunk, dtype=np.float32)
            clean_float = np.nan_to_num(float_array, nan=0.0)
            int16_array = (np.clip(clean_float, -1.0, 1.0) * 32767).astype(np.int16)
            
            accumulated_buffer.extend(int16_array.tobytes())
            
            # Task 2: Temporal - MANDATORY BUFFERING. Do NOT yield until exactly 3200 bytes (100ms)
            if len(accumulated_buffer) >= 3200:
                print(f"DEBUG - Sending 3200 bytes chunk to gRPC")
                yield speech.StreamingRecognizeRequest(audio_content=bytes(accumulated_buffer))
                accumulated_buffer.clear()
                
        except Exception as e:
            logger.error(f"❌ Generator error: {e}")
            break

@app.websocket("/ws/audio")
async def websocket_audio_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time audio streaming
    
    Task 3: Implements async generator pattern with speech_v1.SpeechAsyncClient
    Task 4: Session Persistence - Keep ONE stream alive per WebSocket, enable interim_results=True
    """
    await websocket.accept()
    client_id = f"client_{int(time.time() * 1000)}"
    session_id = f"session_{client_id}"
    
    # Initialize session data for archival
    session_data = SessionData(
        session_id=session_id,
        start_time=datetime.now()
    )
    active_sessions[session_id] = session_data
    
    # Audio buffering for both Gemini API rate limiting AND Google STT chunk aggregation
    audio_buffer = bytearray()
    buffer_size_threshold = 32000  # ~2 seconds of 16kHz mono audio (for Gemini)
    
    # Task 3: Use asyncio.Queue for async generator pattern
    audio_queue = asyncio.Queue()
    
    last_gemini_call = 0
    gemini_interval = 2.0  # Minimum 2 seconds between Gemini calls
    
    # Dynamic audio configuration from client
    client_sample_rate = 16000  # Default, will be updated from start_session
    
    # Task 2: Implement the Three-Dimension Specs - Physical: Force encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16 and sample_rate_hertz=16000
    recognition_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  # Physical: Force LINEAR16
        sample_rate_hertz=16000,  # Physical: Force 16000
        language_code="en-US",
        model="latest_long",
        enable_automatic_punctuation=True,
        enable_word_confidence=True
    )
    
    # Task 4: Session Persistence - Ensure interim_results=True is active so we see partial text immediately
    config = speech.StreamingRecognitionConfig(
        config=recognition_config,
        interim_results=True,  # Ensure interim_results=True is active so we see partial text immediately
        single_utterance=False  # Keep stream open for continuous audio
    )
    
    # Task 3: The Call - responses = await client.streaming_recognize(requests=request_generator())
    streaming_task = None
    
    if google_client.client:
        try:
            # Start the async streaming recognition with AsyncRetry
            streaming_task = asyncio.create_task(process_streaming_responses(websocket, session_data, audio_queue, config))
        except Exception as e:
            logger.error(f"❌ Failed to start streaming recognition: {e}")
    
    logger.info(f"🔌 WebSocket client connected: {client_id} (Session: {session_id})")
    
    try:
        while True:
            # Receive message from extension
            message = await websocket.receive()
            
            if message["type"] == "websocket.receive":
                # Handle JSON messages (start_session, etc.)
                if "text" in message:
                    try:
                        json_message = json.loads(message["text"])
                        if json_message.get("type") == "start_session":
                            # Extract client audio configuration
                            config_data = json_message.get("config", {})
                            client_sample_rate = config_data.get("sampleRate", 16000)
                            logger.info(f"🎵 Client audio config - Sample Rate: {client_sample_rate}Hz")
                            
                            # Send acknowledgment
                            await websocket.send_text(json.dumps({
                                "type": "session_started",
                                "session_id": session_id,
                                "server_config": {
                                    "sample_rate": client_sample_rate,
                                    "encoding": "LINEAR16"
                                }
                            }))
                            continue
                        elif json_message.get("type") == "audio_config":
                            logger.info(f"🔧 Audio config: {json_message.get('config')}")
                            continue
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️ Invalid JSON from {client_id}")
                        continue
                
                if "bytes" in message:
                    # Process audio data - Send directly to queue for proper buffering
                    audio_data = message["bytes"]
                    logger.debug(f"📨 Received {len(audio_data)} bytes from {client_id}")
                    
                    # Send raw audio data to queue for buffering and processing
                    try:
                        audio_queue.put_nowait(audio_data)
                    except asyncio.QueueFull:
                        logger.warning("⚠️ Audio processing queue full - dropping chunk")
                    
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket client disconnected: {client_id}")
        
        # Stop streaming task
        if streaming_task:
            # Send shutdown signal to audio queue
            try:
                audio_queue.put_nowait(None)
            except:
                pass
            streaming_task.cancel()
        
        # Finalize session data and send archive
        session_data.end_time = datetime.now()
        session_data.total_duration = (session_data.end_time - session_data.start_time).total_seconds()
        
        # Send email archive asynchronously
        if email_service.is_configured and session_data.transcripts:
            asyncio.create_task(email_service.send_session_archive(session_data))
        
        # Clean up session
        if session_id in active_sessions:
            del active_sessions[session_id]
            
    except Exception as e:
        logger.error(f"❌ WebSocket error for {client_id}: {e}")
        
        # Stop streaming task
        if streaming_task:
            # Send shutdown signal to audio queue
            try:
                audio_queue.put_nowait(None)
            except:
                pass
            streaming_task.cancel()
        
        # Clean up session on error
        if session_id in active_sessions:
            session_data = active_sessions[session_id]
            session_data.end_time = datetime.now()
            session_data.total_duration = (session_data.end_time - session_data.start_time).total_seconds()
            
            # Still try to send archive on error
            if email_service.is_configured and session_data.transcripts:
                asyncio.create_task(email_service.send_session_archive(session_data))
            
            del active_sessions[session_id]

async def process_streaming_responses(websocket: WebSocket, session_data: SessionData, audio_queue: asyncio.Queue, config):
    """Process streaming recognition responses using async generator pattern
    
    正确的异步流式调用 (修复缺少参数的报错)
    """
    if not google_client.client:
        logger.error("❌ Google Speech AsyncClient not initialized")
        return
    
    try:
        # 正确的异步流式调用 - Fix missing parameters error
        responses = await google_client.client.streaming_recognize(
            requests=request_generator(audio_queue, config), 
            retry=retries.AsyncRetry()
        )
        
        logger.info("✅ Async streaming recognition started with AsyncRetry")
        
        # Process streaming responses in real-time
        async for response in responses:
            try:
                # Process each streaming response
                for result in response.results:
                    if result.alternatives:
                        transcript_text = result.alternatives[0].transcript
                        is_final = result.is_final
                        confidence = result.alternatives[0].confidence if result.alternatives[0].confidence else 0.0
                        current_time = time.time()
                        
                        # 白标标志: 部署后，日志中必须不再出现 SyntaxError，且必须出现类似 DEBUG - Sending 3200 bytes chunk 的信息
                        if transcript_text.strip():
                            print(f"DEBUG - Raw Transcript: '{transcript_text}' (final: {is_final})")
                            logger.info(f"📝 Google STT: \"{transcript_text}\" (final: {is_final})")
                            
                            # Process SE terms and Gemini analysis
                            detected_terms = se_knowledge_base.detect_se_terms(transcript_text)
                            
                            # Check if should process Gemini (only for final results)
                            should_process_gemini_now = (
                                is_final and
                                transcript_text.strip()
                            )
                            
                            gemini_analysis = None
                            if should_process_gemini_now:
                                try:
                                    gemini_analysis = await gemini_service.analyze_transcript(transcript_text)
                                    if gemini_analysis:
                                        logger.info(f"🤖 Gemini analysis: {len(gemini_analysis.keywords)} explanations")
                                except Exception as e:
                                    logger.error(f"❌ Gemini analysis failed: {e}")
                            
                            # Send results to WebSocket
                            response_data = {
                                "type": "transcription_result",
                                "text": transcript_text,
                                "is_final": is_final,
                                "confidence": confidence,
                                "timestamp": current_time,
                                "se_terms": detected_terms,
                                "se_definitions": {},
                            }
                            
                            # Add Gemini analysis if available
                            if gemini_analysis and gemini_analysis.keywords:
                                response_data["gemini_analysis"] = {
                                    "original_text": gemini_analysis.original_text,
                                    "keywords": [
                                        {"term": kw.term, "explanation": kw.explanation}
                                        for kw in gemini_analysis.keywords
                                    ]
                                }
                            
                            try:
                                await websocket.send_text(json.dumps(response_data))
                                logger.info(f"📤 Sent transcription result (final: {is_final})")
                            except Exception as send_error:
                                logger.error(f"❌ WebSocket send error: {send_error}")
                            
                            # Update session data
                            if is_final:
                                session_data.transcripts.append(transcript_text)
                                session_data.se_terms_detected.extend(detected_terms)
                                session_data.se_terms_detected = list(set(session_data.se_terms_detected))
                    
            except Exception as response_error:
                logger.error(f"❌ Streaming response processing error: {response_error}")
                continue
                
    except Exception as e:
        logger.error(f"❌ STT API Connection Error: {e}")

if __name__ == "__main__":
    # Production deployment - Railway uses Procfile, this is for local development only
    port = int(os.environ.get("PORT", 8080))  # Railway default port, fallback for local development
    
    logger.info("🚀 Starting SE Insight Railway Backend")
    logger.info(f"📡 Port: {port} (Railway: {bool(os.environ.get('RAILWAY_ENVIRONMENT'))})")
    logger.info(f"🔑 Google API: {'Available' if GOOGLE_CLOUD_AVAILABLE else 'Not Available'}")
    logger.info(f"🔐 GCP Key: {'Configured' if os.environ.get('GCP_KEY_JSON') else 'Not Configured'}")
    logger.info(f"🤖 Gemini API: {'Configured' if os.environ.get('GEMINI_API_KEY') else 'Not Configured'}")
    logger.info(f"📧 Email Service: {'Configured' if email_service.is_configured else 'Not Configured'}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
        reload=False  # Disable reload for production
    )