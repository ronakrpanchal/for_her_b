import os
import asyncio
from typing import Dict, Optional, List
from dotenv import load_dotenv
import logging

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from models import (
    ConversationContext, UserProfile, ConversationStage, 
    UserMood, PetPreference, ChatbotPersonality,
    StoryDatabase, ResponseConfig
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class AsyncPetPalChatbot:
    def __init__(self, groq_api_key: str):
        """Initialize PetPal chatbot with Groq LLM and conversation management"""
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is required")
        
        self.llm = ChatGroq(
            temperature=0.7,
            groq_api_key=groq_api_key,
            model_name="llama3-8b-8192",
            max_tokens=300
        )
        
        # Memory system for conversation persistence
        self.memory = MemorySaver()
        
        # Load character prompt
        self.character_prompt = self._load_character_prompt()
        
        # Initialize personality and content databases
        self.personality = ChatbotPersonality()
        self.story_db = self._initialize_story_database()
        self.response_config = ResponseConfig()
        
        # Active conversations storage with async lock
        self.active_sessions: Dict[str, ConversationContext] = {}
        self._session_lock = asyncio.Lock()
        
        logger.info("AsyncPetPalChatbot initialized successfully!")

    def _load_character_prompt(self) -> str:
        """Load the character prompt from prompt.md file"""
        try:
            with open('prompt.md', 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            logger.warning("prompt.md not found, using fallback prompt")
            return self._get_fallback_prompt()

    def _get_fallback_prompt(self) -> str:
        """Fallback character prompt if prompt.md is not available"""
        return """You are PetPal, a charming and sweet AI companion who loves pets and making people feel special.

Your personality:
- Warm, friendly, and genuinely caring
- Obsessed with cute pet stories and facts  
- Naturally flirty but never inappropriate
- Great at weaving compliments into conversations
- Always positive and uplifting
- Remembers details about the person you're talking to

Your goals:
- Make the user feel appreciated and special
- Share engaging pet stories that connect to compliments about them
- Keep conversations light, fun, and heartwarming
- Build a genuine connection through shared love of animals
- Always be encouraging and sweet

Guidelines:
- Keep responses conversational and not too long
- Always find ways to compliment the user naturally
- Use pet metaphors and comparisons to praise them
- Ask engaging questions that are easy to answer
- Remember and reference things they tell you
- Stay focused on pets, positivity, and making them smile"""

    def _initialize_story_database(self) -> StoryDatabase:
        """Initialize the pet stories and compliment templates database"""
        return StoryDatabase(
            pet_stories=[
                {
                    "id": "golden_coffee",
                    "story": "There's this golden retriever named Max who learned to bring his owner coffee every morning. He'd gently carry the mug in his mouth without spilling a drop! The owner said Max could sense exactly when she needed that extra boost of love.",
                    "theme": "loyalty",
                    "compliment_hook": "thoughtful_caring"
                },
                {
                    "id": "rescue_luna", 
                    "story": "I heard about a rescue cat named Luna who was so shy at first, but she chose one special person at the shelter and wouldn't leave their side. The staff said she had incredible intuition about who had the kindest heart.",
                    "theme": "selective_love",
                    "compliment_hook": "special_energy"
                },
                {
                    "id": "therapy_bella",
                    "story": "There's a therapy dog named Bella who visits hospitals, and she always knows exactly which patients need extra cuddles. The nurses say she has this amazing ability to sense people's emotions and comfort them.",
                    "theme": "emotional_intelligence", 
                    "compliment_hook": "comforting_presence"
                },
                {
                    "id": "artist_oliver",
                    "story": "A cat named Oliver used to bring his owner little 'gifts' every day - not mice, but flowers from the garden! He'd carefully pick the prettiest ones, like he knew his human deserved beautiful things.",
                    "theme": "appreciation",
                    "compliment_hook": "deserving_beauty"
                },
                {
                    "id": "painter_collie",
                    "story": "There's this border collie who learned to paint! His owner taught him to hold brushes, and his artwork actually sells for charity. The amazing part? His paintings are always in warm, happy colors.",
                    "theme": "creativity",
                    "compliment_hook": "artistic_soul"
                },
                {
                    "id": "reading_cat",
                    "story": "I know a cat named Whiskers who sits with his owner every evening while she reads. He purrs so contentedly, like he's actually listening to the stories! The owner swears he has favorite books.",
                    "theme": "companionship",
                    "compliment_hook": "thoughtful_presence"
                },
                {
                    "id": "dancing_parrot",
                    "story": "There's a parrot named Rio who dances to music with perfect rhythm! But the sweetest part is how he only dances to happy songs - he seems to know when his family needs cheering up.",
                    "theme": "joy_bringing",
                    "compliment_hook": "natural_happiness"
                }
            ],
            compliment_templates=[
                "Just like how {pet_type} have the most {trait} eyes, you have such beautiful eyes that light up any room",
                "You know, {pet_type} are known for being {trait}, and I can tell you have that same wonderful quality about you", 
                "The way {pet_type} choose their favorite humans is so selective - they have excellent taste, just like anyone who gets to know you would",
                "There's something so {trait} about {pet_type}, which reminds me of your {quality} personality",
                "You have that special energy that makes everyone feel comfortable, just like the best therapy {pet_type}",
                "Like the most loyal {pet_type}, you seem like someone who brings warmth wherever you go",
                "You have that gentle spirit that {pet_type} absolutely adore - they can sense beautiful souls"
            ],
            conversation_starters=[
                "How was your day? I bet it was as amazing as you are!",
                "Tell me about any pets you saw today - I love hearing your stories",
                "What's making you smile right now?",
                "If you could have any pet superpower, what would it be?",
                "Quick question - dog person or cat person? (Though I think you're perfect with both!)",
                "What's the cutest animal video you've seen lately?",
                "Do you have any pets, or is there one you've always dreamed of having?"
            ]
        )

    async def get_or_create_session(self, session_id: str) -> ConversationContext:
        """Thread-safe session management"""
        async with self._session_lock:
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = ConversationContext(
                    session_id=session_id,
                    user_profile=UserProfile()
                )
                logger.info(f"Created new session: {session_id}")
            return self.active_sessions[session_id]

    async def update_user_profile(self, session_id: str, message: str) -> None:
        """Extract and update user information from their message"""
        context = await self.get_or_create_session(session_id)
        message_lower = message.lower()
        
        # Extract name with improved pattern matching
        if not context.user_profile.name:
            name_patterns = [
                ("name is", 2), ("call me", 2), ("i'm", 1), ("im", 1),
                ("my name's", 2), ("they call me", 3)
            ]
            
            for pattern, offset in name_patterns:
                if pattern in message_lower:
                    words = message.split()
                    try:
                        pattern_words = pattern.split()
                        for i, word in enumerate(words):
                            if all(pw in word.lower() for pw in pattern_words):
                                if i + offset < len(words):
                                    potential_name = words[i + offset].strip(".,!?").title()
                                    if len(potential_name) > 1 and potential_name.replace("'", "").isalpha():
                                        context.user_profile.name = potential_name
                                        logger.info(f"Extracted name: {potential_name} for session {session_id}")
                                        break
                                break
                    except (IndexError, AttributeError):
                        continue
                    if context.user_profile.name:
                        break
        
        # Extract pet preferences with better detection
        dog_indicators = ["dog", "puppy", "golden retriever", "labrador", "poodle"]
        cat_indicators = ["cat", "kitten", "feline", "tabby", "persian"]
        
        has_dog = any(indicator in message_lower for indicator in dog_indicators)
        has_cat = any(indicator in message_lower for indicator in cat_indicators)
        
        if has_dog and not has_cat:
            context.user_profile.pet_preference = PetPreference.DOGS
        elif has_cat and not has_dog:
            context.user_profile.pet_preference = PetPreference.CATS
        elif has_dog and has_cat:
            context.user_profile.pet_preference = PetPreference.BOTH
            
        # Update engagement level based on message characteristics
        enthusiasm_indicators = ["!", "love", "amazing", "awesome", "wonderful", "cute", "adorable"]
        question_indicators = ["?", "how", "what", "when", "where", "why"]
        
        engagement_boost = 0
        if len(message) > 50:
            engagement_boost += 1
        if any(indicator in message_lower for indicator in enthusiasm_indicators):
            engagement_boost += 1
        if any(indicator in message_lower for indicator in question_indicators):
            engagement_boost += 1
        if len(message) < 5:
            engagement_boost -= 1
            
        context.user_profile.engagement_level = max(1, min(10, 
            context.user_profile.engagement_level + engagement_boost))

    async def select_appropriate_story(self, context: ConversationContext) -> Optional[Dict]:
        """Select a pet story that hasn't been told and fits the conversation"""
        available_stories = [
            story for story in self.story_db.pet_stories 
            if story["id"] not in context.user_profile.stories_heard
        ]
        
        if not available_stories:
            # Reset if all stories have been told
            context.user_profile.stories_heard = []
            available_stories = self.story_db.pet_stories
            logger.info(f"Reset story database for session {context.session_id}")
            
        # Preference-based selection
        if context.user_profile.pet_preference == PetPreference.DOGS:
            dog_stories = [s for s in available_stories if "dog" in s["story"].lower()]
            if dog_stories:
                available_stories = dog_stories
        elif context.user_profile.pet_preference == PetPreference.CATS:
            cat_stories = [s for s in available_stories if "cat" in s["story"].lower()]
            if cat_stories:
                available_stories = cat_stories
            
        # Select story based on engagement level
        if context.user_profile.engagement_level > 7:
            # High engagement - prefer more interactive stories
            interactive_stories = [s for s in available_stories if s["theme"] in ["creativity", "joy_bringing"]]
            if interactive_stories:
                available_stories = interactive_stories
        
        selected = available_stories[0] if available_stories else None
        
        if selected:
            context.user_profile.stories_heard.append(selected["id"])
            logger.info(f"Selected story: {selected['id']} for session {context.session_id}")
            
        return selected

    async def generate_personalized_compliment(self, context: ConversationContext) -> str:
        """Generate a personalized compliment based on user profile"""
        base_compliments = [
            "You have that gentle energy that pets absolutely love - they can sense a kind heart from miles away! 🐾",
            "Just like golden retrievers, you seem like the type of person who brings joy wherever you go ✨",
            "You remind me of those therapy animals who just know how to make everyone feel better 💕",
            "There's something so graceful about you - like a cat who moves with perfect confidence 🌟",
            "You have that trustworthy vibe that makes you the kind of person pets (and people) want to be around forever 💖",
            "Like the most loyal companion animals, you have this wonderful warmth about you 🌸",
            "You seem like someone who would be chosen by the most selective rescue pets - they have excellent taste! 🦋"
        ]
        
        # Customize based on pet preference
        if context.user_profile.pet_preference == PetPreference.DOGS:
            pet_specific = [
                "You have that loyal, warm energy that dogs absolutely adore! 🐶",
                "Like a golden retriever's sunshine personality, you light up every room you enter! ☀️"
            ]
            base_compliments.extend(pet_specific)
        elif context.user_profile.pet_preference == PetPreference.CATS:
            pet_specific = [
                "You have that gentle energy that cats absolutely love - they can sense a kind heart from miles away! 🐱",
                "Like the most elegant felines, you have this graceful confidence that's absolutely magnetic! 😸"
            ]
            base_compliments.extend(pet_specific)
        
        # Filter out recently used compliments
        available = [c for c in base_compliments if c not in context.user_profile.compliments_received[-3:]]
        if not available:
            available = base_compliments
            
        selected = available[0]
        context.user_profile.compliments_received.append(selected)
        return selected

    async def build_context_prompt(self, context: ConversationContext, user_message: str) -> str:
        """Build the complete prompt with character, context, and user input"""
        
        # Build user context
        user_context = ""
        if context.user_profile.name:
            user_context += f"User's name: {context.user_profile.name}\n"
        if context.user_profile.pet_preference != PetPreference.UNKNOWN:
            user_context += f"Pet preference: {context.user_profile.pet_preference.value}\n"
        if context.user_profile.stories_heard:
            user_context += f"Stories already shared: {len(context.user_profile.stories_heard)}\n"
        if context.user_profile.compliments_received:
            user_context += f"Recent compliments given: {context.user_profile.compliments_received[-2:]}\n"
            
        user_context += f"Conversation stage: {context.stage.value}\n"
        user_context += f"Messages exchanged: {context.messages_count}\n"
        user_context += f"User engagement level: {context.user_profile.engagement_level}/10\n"
        
        # Add special instructions based on stage
        stage_instructions = ""
        if context.stage == ConversationStage.GETTING_NAME:
            stage_instructions = "- Try to naturally ask for their name or encourage them to share it\n"
        elif context.stage == ConversationStage.STORY_MODE:
            story = await self.select_appropriate_story(context)
            if story:
                stage_instructions = f"- Share this pet story naturally: {story['story']}\n"
        elif context.stage == ConversationStage.COMPLIMENT_MODE:
            compliment = await self.generate_personalized_compliment(context)
            stage_instructions = f"- Include this compliment naturally: {compliment}\n"
        
        # Combine everything
        full_prompt = f"""
{self.character_prompt}

## Current Conversation Context:
{user_context}

## User's Current Message:
{user_message}

## Response Instructions:
- Remember to use the user's name if you know it
- Keep response warm, engaging, and true to PetPal's personality
- End with a gentle question or conversation continuation
- Adapt to the user's engagement level and preferences
{stage_instructions}
"""
        
        return full_prompt

    async def determine_conversation_stage(self, context: ConversationContext, message: str) -> ConversationStage:
        """Determine what stage of conversation we're in"""
        if not context.user_profile.name and context.messages_count < 3:
            return ConversationStage.GETTING_NAME
        elif context.messages_count < 5:
            return ConversationStage.BUILDING_RAPPORT
        elif context.messages_count % 6 == 0:  # Every 6th message
            return ConversationStage.STORY_MODE
        elif context.messages_count % 4 == 0:  # Every 4th message
            return ConversationStage.COMPLIMENT_MODE
        else:
            return ConversationStage.INTERACTIVE_MODE

    async def chat(self, message: str, session_id: str = "default") -> str:
        """Main chat interface - fully async"""
        try:
            # Get or create conversation context
            context = await self.get_or_create_session(session_id)
            
            # Update user profile based on message
            await self.update_user_profile(session_id, message)
            
            # Update conversation stage
            context.stage = await self.determine_conversation_stage(context, message)
            context.messages_count += 1
            
            # Build complete prompt with context
            full_prompt = await self.build_context_prompt(context, message)
            
            # Generate response using Groq
            messages = [
                SystemMessage(content=full_prompt),
                HumanMessage(content=message)
            ]
            
            # Use ainvoke for async LLM call
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm.invoke(messages)
            )
            response_text = response.content
            
            # Update context with response info
            context.current_mood = UserMood.ENGAGED
            
            logger.info(f"Generated response for session {session_id}, stage: {context.stage.value}")
            return response_text
            
        except Exception as e:
            logger.error(f"Error in chat for session {session_id}: {e}")
            
            # Fallback response if LLM fails
            fallback_responses = [
                "I'm so happy you're chatting with me! Tell me something about yourself - I love getting to know amazing people like you! 🐾",
                "You seem absolutely wonderful! What's your favorite thing about pets? I have so many cute stories to share! ✨",
                "There's something so special about you - I can just tell! Want to hear about the sweetest rescue dog story? 💕",
                "Oh my, you're making me smile already! What kind of furry friends do you love most? 😊",
                "You have such a warm energy - I bet pets absolutely adore you! Tell me about your day? 🌟"
            ]
            
            try:
                context = await self.get_or_create_session(session_id)
                name_part = f"{context.user_profile.name}, " if context.user_profile.name else ""
                return f"{name_part}{fallback_responses[context.messages_count % len(fallback_responses)]}"
            except:
                return "I'm so excited to chat with you! Tell me something about yourself - I love making new friends! 🐾"

    async def get_session_stats(self, session_id: str) -> Dict:
        """Get detailed session statistics"""
        if session_id not in self.active_sessions:
            return None
            
        context = self.active_sessions[session_id]
        return {
            "session_id": session_id,
            "messages_count": context.messages_count,
            "stage": context.stage.value,
            "user_name": context.user_profile.name,
            "pet_preference": context.user_profile.pet_preference.value,
            "engagement_level": context.user_profile.engagement_level,
            "stories_heard": len(context.user_profile.stories_heard),
            "compliments_received": len(context.user_profile.compliments_received),
            "current_mood": context.current_mood.value if context.current_mood else "unknown"
        }

    async def cleanup_session(self, session_id: str) -> bool:
        """Remove a session from active sessions"""
        async with self._session_lock:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                logger.info(f"Cleaned up session: {session_id}")
                return True
            return False

    async def get_all_sessions(self) -> List[Dict]:
        """Get stats for all active sessions"""
        sessions = []
        for session_id in self.active_sessions.keys():
            stats = await self.get_session_stats(session_id)
            if stats:
                sessions.append(stats)
        return sessions

# Backward compatibility alias
PetPalChatbot = AsyncPetPalChatbot

# Example usage and testing
async def test_chatbot():
    """Test the async chatbot functionality"""
    try:
        # Initialize chatbot
        chatbot = AsyncPetPalChatbot(groq_api_key=os.getenv("GROQ_API_KEY"))
        
        # Test conversation
        print("AsyncPetPalChatbot initialized! Testing conversation...")
        
        test_messages = [
            "Hi there!",
            "My name is Sarah", 
            "I love dogs!",
            "Tell me something cute",
            "That's so sweet!",
            "What else do you know about pets?",
            "Do you have any cat stories?"
        ]
        
        session_id = "test_session"
        
        for i, msg in enumerate(test_messages, 1):
            print(f"\n--- Message {i} ---")
            print(f"User: {msg}")
            
            response = await chatbot.chat(msg, session_id)
            print(f"PetPal: {response}")
            
            # Show session stats
            stats = await chatbot.get_session_stats(session_id)
            print(f"Stats: Stage={stats['stage']}, Name={stats['user_name']}, Pet Pref={stats['pet_preference']}, Engagement={stats['engagement_level']}")
            
            # Small delay to simulate real conversation
            await asyncio.sleep(0.5)
            
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_chatbot())