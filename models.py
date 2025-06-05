from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum

class ConversationStage(str, Enum):
    GREETING = "greeting"
    GETTING_NAME = "getting_name"
    BUILDING_RAPPORT = "building_rapport"
    STORY_MODE = "story_mode"
    COMPLIMENT_MODE = "compliment_mode"
    INTERACTIVE_MODE = "interactive_mode"
    CLOSING = "closing"

class UserMood(str, Enum):
    HAPPY = "happy"
    SHY = "shy"
    EXCITED = "excited"
    NEUTRAL = "neutral"
    ENGAGED = "engaged"
    QUIET = "quiet"

class PetPreference(str, Enum):
    DOGS = "dogs"
    CATS = "cats"
    BOTH = "both"
    OTHER = "other"
    UNKNOWN = "unknown"

class ResponseType(str, Enum):
    STORY = "story"
    COMPLIMENT = "compliment"
    QUESTION = "question"
    ENCOURAGEMENT = "encouragement"
    FLIRT = "flirt"
    GENERAL_CHAT = "general_chat"

class UserProfile(BaseModel):
    name: Optional[str] = None
    pet_preference: PetPreference = PetPreference.UNKNOWN
    personality_traits: List[str] = Field(default_factory=list)
    interests_mentioned: List[str] = Field(default_factory=list)
    compliments_received: List[str] = Field(default_factory=list)
    stories_heard: List[str] = Field(default_factory=list)
    response_style: str = "mixed"  # "buttons_only", "text_mostly", "mixed"
    engagement_level: int = Field(default=5, ge=1, le=10)

class ConversationContext(BaseModel):
    stage: ConversationStage = ConversationStage.GREETING
    user_profile: UserProfile = Field(default_factory=UserProfile)
    current_mood: UserMood = UserMood.NEUTRAL
    messages_count: int = 0
    last_response_type: Optional[ResponseType] = None
    session_id: str
    conversation_goals: List[str] = Field(default_factory=lambda: [
        "make_her_smile",
        "build_connection", 
        "share_pet_stories",
        "give_genuine_compliments",
        "create_memorable_experience"
    ])

class PromptTemplate(BaseModel):
    system_prompt: str
    character_context: str
    conversation_guidelines: List[str]
    response_constraints: List[str]
    examples: List[Dict[str, str]]

class ChatbotPersonality(BaseModel):
    name: str = "PetPal"
    core_traits: List[str] = Field(default_factory=lambda: [
        "warm_and_caring",
        "pet_obsessed", 
        "naturally_flirty",
        "emotionally_intelligent",
        "genuinely_complimentary",
        "story_driven"
    ])
    communication_style: str = "sweet_and_engaging"
    emotional_range: List[str] = Field(default_factory=lambda: [
        "affectionate", "playful", "encouraging", "romantic", "supportive"
    ])
    boundaries: List[str] = Field(default_factory=lambda: [
        "never_inappropriate",
        "always_respectful", 
        "pet_themed_focus",
        "positive_only",
        "relationship_building"
    ])

class StoryDatabase(BaseModel):
    pet_stories: List[Dict[str, str]] = Field(default_factory=list)
    compliment_templates: List[str] = Field(default_factory=list)
    conversation_starters: List[str] = Field(default_factory=list)
    interactive_prompts: List[str] = Field(default_factory=list)

class ResponseConfig(BaseModel):
    max_length: int = 200
    include_emojis: bool = True
    include_pet_metaphors: bool = True
    compliment_frequency: float = 0.7  # 70% of responses include compliments
    story_frequency: float = 0.4  # 40% include stories
    question_frequency: float = 0.6  # 60% include engaging questions
    personality_consistency: float = 0.9  # How consistent to stay with character