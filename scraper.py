import wikipedia
import re
import nltk
from typing import List, Dict

# Optional: download NLTK resources once
nltk.download('punkt')

def clean_text(text: str) -> str:
    """Clean and simplify raw Wikipedia text."""
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_relevant_sections(text: str, keywords: List[str], max_sections: int = 3) -> str:
    """Extract relevant sections that mention specific keywords like 'Personality'."""
    sections = re.split(r'==+\s*(.*?)\s*==+', text)
    extracted = []

    # sections[0] is the intro, then alternating section title and content
    for i in range(1, len(sections)-1, 2):
        title = sections[i].lower()
        content = sections[i+1].strip()
        if any(k.lower() in title for k in keywords) and len(content) > 100:
            extracted.append(f"## {sections[i]} ##\n{content}")
            if len(extracted) >= max_sections:
                break
    return "\n\n".join(extracted)

def fetch_wikipedia_summary(title: str) -> str:
    """Fetch and clean Wikipedia content for a given title."""
    try:
        content = wikipedia.page(title).content
    except wikipedia.exceptions.PageError:
        print(f"[!] Page not found for: {title}")
        return ""

    cleaned = clean_text(content)
    intro = summarize_paragraphs(cleaned)
    extra = extract_relevant_sections(content, ["personality", "character", "psychology", "persona", "traits", "analysis"])
    return f"{intro}\n\n{extra}"

def summarize_paragraphs(text: str, max_paragraphs: int = 5) -> str:
    """Return the top n non-empty paragraphs from the text."""
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 50]
    return "\n\n".join(paragraphs[:max_paragraphs])

def scrape_characters(titles: List[str]) -> Dict[str, str]:
    """Scrape summaries for multiple character titles."""
    results = {}
    for title in titles:
        print(f"[+] Scraping: {title}")
        summary = fetch_wikipedia_summary(title)
        if summary:
            results[title] = summary
    return results

if __name__ == "__main__":
    character_list = [
    "Gilgamesh",
    "Enkidu",
    "Ishtar",
    "Ea (god)",  # Mesopotamian god
    "Utnapishtim",
    "Socrates (Philosophy teacher of Plato)",
    "Glaucon (brother of Plato)",
    "Adeimantus of Collytus",
    "Medea (mythology)",
    "Jason (mythology)",
    "Aegeus",
    "Erinyes (Furies, Goddess of vengance)",
    "Apollo",
    "Daphne (mythology)",
    "Lesbia (real name Clodia, Lover of Catullus)",  # Lover of Catullus
    "Gallis (mythology)",  # Refers to Galli priests, assuming typo or alt ref
    "Dante Alighieri",
    "Virgil",
    "Beatrice Portinari (Dantes Muse)",  # Dante’s muse
    "Homer",
    "Horace",
    "Sugawara no Michizane",  # Full historical name for Tachibana no Michizane
    "Scheherazade",
    "Shahryar (a character in One Thousand and One Nights)",  # King Shahryar simplified
    "Aladdin",
    "Dinarzade",
    "Michel de Montaigne",
    "Niccolò Machiavelli",
    "Richard II (Shakespeare play)",
    "King Henry IV (Shakespeare play)",
]



    scraped_data = scrape_characters(character_list)

    # Save to file for now
    with open("character_wikis.txt", "w", encoding="utf-8") as f:
        for name, text in scraped_data.items():
            f.write(f"### {name} ###\n{text}\n\n")
