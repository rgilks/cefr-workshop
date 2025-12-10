"""
Generate synthetic sample data for testing the CEFR training pipeline.

This script creates fake essays at each CEFR level so you can test the full 
pipeline BEFORE obtaining the real Write & Improve corpus from Cambridge.

Usage:
    python generate_sample_data.py
    
This will create:
    data/train.jsonl  (40 samples)
    data/dev.jsonl    (5 samples)
    data/test.jsonl   (5 samples)

⚠️  These synthetic essays are for TESTING ONLY. Train on real data for
    production models. The W&I corpus is available from:
    https://www.cl.cam.ac.uk/research/nl/bea2019st/#data
"""
import json
import random
from pathlib import Path

# Seed for reproducibility
random.seed(42)

# Sample sentence templates by CEFR level
# These demonstrate characteristic linguistic features at each level

TEMPLATES = {
    "A1": [
        # Very simple present tense, basic vocabulary
        "I am {name}. I like {food}. {food} is good.",
        "My name is {name}. I live in {city}. I am {age} years old.",
        "I have {pet}. My {pet} is {color}. I love my {pet}.",
        "Today is {day}. The weather is {weather}. I am happy.",
        "I go to school. I like {subject}. My teacher is nice.",
    ],
    "A2": [
        # Simple past, basic connectors
        "Yesterday I went to {place}. I saw many {things}. It was very {adj}.",
        "I like {activity} because it is fun. I do it every {day}.",
        "My favorite {category} is {item}. I think it is {adj} and {adj2}.",
        "Last week I visited my {relative}. We ate {food} together. It was nice.",
        "I want to learn {language} because I want to visit {country}.",
    ],
    "B1": [
        # More complex sentences, opinions, some conditionals
        "In my opinion, {topic} is becoming more important in today's society. "
        "Many people believe that {opinion}. However, I think we should consider {alternative}.",
        
        "If I had more time, I would like to {activity}. "
        "Currently, I spend most of my time {current_activity}, which is also enjoyable.",
        
        "The main advantage of {thing} is that it allows us to {benefit}. "
        "On the other hand, some people argue that {disadvantage}.",
        
        "When I was younger, I used to {past_habit}. Now I prefer {current_habit} "
        "because I find it more {adj} and {adj2}.",
    ],
    "B2": [
        # Complex structures, varied vocabulary, cohesive arguments
        "While it is undeniable that {topic} has transformed our daily lives, "
        "we must carefully consider both the benefits and drawbacks of this development. "
        "On one hand, {benefit} has enabled us to {positive_outcome}. "
        "Conversely, there are legitimate concerns regarding {concern}.",
        
        "The question of whether {debate_topic} remains highly contentious. "
        "Proponents argue that {pro_argument}, citing evidence from {source}. "
        "Nevertheless, critics point out that {counter_argument}, "
        "which suggests a more nuanced approach may be necessary.",
        
        "Having considered the various perspectives on {issue}, "
        "I am inclined to support the view that {position}. "
        "This conclusion is based on {reason1} and {reason2}, "
        "although I acknowledge that {concession}.",
    ],
    "C1": [
        # Sophisticated structures, idiomatic expressions, nuanced arguments
        "It would be naive to assume that {topic} can be addressed through "
        "simplistic solutions. The underlying complexity of this issue stems from "
        "a multitude of interconnected factors, including {factor1}, {factor2}, "
        "and the often-overlooked dimension of {factor3}. "
        "What is particularly striking is the extent to which {observation}.",
        
        "The prevailing discourse surrounding {topic} tends to overlook "
        "a crucial aspect: namely, that {overlooked_point}. "
        "By taking this into account, we can develop a more sophisticated "
        "understanding of why {phenomenon} occurs and, more importantly, "
        "what measures might prove effective in addressing it.",
        
        "Whilst proponents of {viewpoint} make compelling arguments, "
        "I would contend that their analysis fails to account for {oversight}. "
        "A more rigorous examination reveals that {deeper_insight}, "
        "which has profound implications for how we approach {application}.",
    ],
    "C2": [
        # Masterful command, precise vocabulary, subtle argumentation
        "The epistemological underpinnings of {topic} warrant considerably more "
        "scrutiny than they typically receive in mainstream discourse. "
        "Indeed, one might argue that the very framing of the debate presupposes "
        "certain ontological commitments that, upon closer examination, prove to be "
        "philosophically untenable. What emerges from a more rigorous analysis is "
        "a picture of remarkable complexity, in which {nuanced_observation}.",
        
        "To characterise {phenomenon} as merely {simple_characterisation} would be "
        "to fundamentally misapprehend its nature. The phenomenon in question is, "
        "in essence, a manifestation of deeper structural dynamics that permeate "
        "{broader_context}. It is only by situating our analysis within this "
        "broader framework that we can hope to grasp {key_insight}.",
        
        "The extent to which {topic} has been instrumentalised in contemporary "
        "discourse is, I would suggest, symptomatic of a more pervasive tendency "
        "to privilege {tendency1} over {tendency2}. This reductionist approach, "
        "however intellectually expedient, obscures the fundamental tensions "
        "inherent in {underlying_tension}.",
    ],
}

# Fill-in values for templates
FILL_VALUES = {
    "name": ["Tom", "Sara", "Alex", "Maria", "John", "Emma"],
    "food": ["pizza", "pasta", "rice", "bread", "cake", "soup"],
    "city": ["London", "Paris", "Tokyo", "New York", "Sydney"],
    "age": ["20", "25", "30", "18", "35"],
    "pet": ["dog", "cat", "bird", "fish", "rabbit"],
    "color": ["brown", "black", "white", "gray", "golden"],
    "day": ["Monday", "Friday", "Sunday", "Wednesday"],
    "weather": ["sunny", "rainy", "cloudy", "nice", "warm"],
    "subject": ["English", "math", "science", "art", "music"],
    "place": ["the park", "a museum", "the beach", "a restaurant"],
    "things": ["people", "animals", "flowers", "buildings", "trees"],
    "adj": ["interesting", "beautiful", "important", "useful", "enjoyable"],
    "adj2": ["practical", "valuable", "memorable", "challenging", "rewarding"],
    "activity": ["reading", "traveling", "cooking", "learning languages", "playing sports"],
    "relative": ["grandmother", "uncle", "cousin", "parents", "friends"],
    "category": ["movie", "book", "sport", "hobby", "subject"],
    "item": ["action films", "mystery novels", "football", "photography", "history"],
    "language": ["Spanish", "French", "Japanese", "German", "Chinese"],
    "country": ["Spain", "France", "Japan", "Germany", "Brazil"],
    "topic": ["technology", "education", "climate change", "social media", "globalization"],
    "opinion": ["it brings more benefits than drawbacks", "change is necessary"],
    "alternative": ["the potential risks", "long-term consequences", "different perspectives"],
    "thing": ["modern technology", "online education", "remote work"],
    "benefit": ["increased efficiency", "greater accessibility", "cost reduction"],
    "positive_outcome": ["communicate more effectively", "access information instantly"],
    "concern": ["privacy issues", "environmental impact", "social inequality"],
    "current_activity": ["working", "studying", "developing my skills"],
    "past_habit": ["play outside all day", "read comic books", "watch cartoons"],
    "current_habit": ["exercising regularly", "reading non-fiction", "meditation"],
    "debate_topic": ["technology improves our lives", "education should be reformed"],
    "pro_argument": ["increased productivity and efficiency support this view"],
    "source": ["recent studies", "empirical research", "historical precedent"],
    "counter_argument": ["the methodology of such studies may be flawed"],
    "issue": ["balancing work and personal life", "environmental sustainability"],
    "position": ["a balanced approach is essential", "gradual reform is preferable"],
    "reason1": ["practical considerations", "empirical evidence"],
    "reason2": ["ethical principles", "long-term sustainability"],
    "concession": ["circumstances may vary", "exceptions certainly exist"],
    "factor1": ["economic pressures", "political considerations"],
    "factor2": ["cultural dynamics", "technological constraints"],
    "factor3": ["historical context", "psychological factors"],
    "observation": ["conventional wisdom often fails to capture this complexity"],
    "overlooked_point": ["implementation challenges are often underestimated"],
    "phenomenon": ["resistance to change manifests", "unintended consequences emerge"],
    "viewpoint": ["progressive reform", "conservative preservation"],
    "oversight": ["practical implementation constraints", "unintended consequences"],
    "deeper_insight": ["systemic factors play a decisive role"],
    "application": ["policy development", "strategic planning"],
    "simple_characterisation": ["a technical challenge", "an economic issue"],
    "broader_context": ["contemporary social structures", "global power dynamics"],
    "key_insight": ["the interplay between agency and structure"],
    "nuanced_observation": ["apparent contradictions often dissolve upon closer inspection"],
    "tendency1": ["quantitative metrics", "short-term outcomes"],
    "tendency2": ["qualitative understanding", "long-term implications"],
    "underlying_tension": ["individual autonomy and collective welfare"],
}


def fill_template(template: str) -> str:
    """Fill a template with random values."""
    result = template
    for key, values in FILL_VALUES.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    return result


def generate_essay(cefr_level: str) -> str:
    """Generate a synthetic essay for a given CEFR level."""
    templates = TEMPLATES[cefr_level]
    
    # Select 2-4 templates based on level complexity
    if cefr_level in ["A1", "A2"]:
        num_paragraphs = random.randint(2, 3)
    elif cefr_level in ["B1", "B2"]:
        num_paragraphs = random.randint(2, 3)
    else:  # C1, C2
        num_paragraphs = random.randint(1, 2)
    
    paragraphs = []
    for _ in range(num_paragraphs):
        template = random.choice(templates)
        paragraphs.append(fill_template(template))
    
    return " ".join(paragraphs)


def main():
    """Generate sample data files."""
    # CEFR to numeric score
    cefr_to_score = {
        "A1": 1.0, "A2": 2.0, "B1": 3.0,
        "B2": 4.0, "C1": 5.0, "C2": 6.0,
    }
    
    # Generate essays - distribute across levels
    # (fewer A1 and C2 as they're rare in real data)
    level_counts = {
        "A1": 4, "A2": 12, "B1": 14,
        "B2": 12, "C1": 6, "C2": 2,
    }
    
    all_essays = []
    for cefr, count in level_counts.items():
        for i in range(count):
            essay = generate_essay(cefr)
            all_essays.append({
                "input": essay,
                "target": cefr_to_score[cefr],
                "cefr": cefr,  # Keep for reference
            })
    
    random.shuffle(all_essays)
    
    # Split: 40 train, 5 dev, 5 test
    train = all_essays[:40]
    dev = all_essays[40:45]
    test = all_essays[45:50]
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Write files
    def write_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                # Only include input and target (what training expects)
                f.write(json.dumps({
                    "input": item["input"],
                    "target": item["target"],
                }, ensure_ascii=False) + "\n")
    
    write_jsonl(train, data_dir / "train.jsonl")
    write_jsonl(dev, data_dir / "dev.jsonl")
    write_jsonl(test, data_dir / "test.jsonl")
    
    print("✅ Generated sample data!")
    print(f"   data/train.jsonl: {len(train)} samples")
    print(f"   data/dev.jsonl:   {len(dev)} samples")
    print(f"   data/test.jsonl:  {len(test)} samples")
    print()
    print("📊 CEFR distribution in training set:")
    from collections import Counter
    cefr_counts = Counter(e["cefr"] for e in train)
    for cefr in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        print(f"   {cefr}: {cefr_counts.get(cefr, 0)}")
    print()
    print("⚠️  Remember: This is SYNTHETIC data for testing only!")
    print("   For real training, obtain the W&I corpus from Cambridge.")


if __name__ == "__main__":
    main()
