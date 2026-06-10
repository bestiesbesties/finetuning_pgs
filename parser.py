from tika import parser
import re
import spacy
nlp = spacy.load("nl_core_news_lg") 

class Parser():
    def __init__(self):
        print("loaded scanner")

    def __call__(self, fp:str):
        context = parser.from_file(fp)
        text = context["content"].replace("\n", "")
        assert len(text) > 0
        chunks = self.chunker2(text)
        return chunks

    
    def chunker2(self, text, max_words=25):
        ## clean
        cleaned = re.sub(r"[^A-Za-z\s.,'\"()\-/]", "", text)
        stripped = re.sub(r"\s+", " ", cleaned)

        ## gather
        sentences = [sent.text for sent in nlp(stripped).sents]
        
        ## aggrate
        combined = []
        buffer = []
        for sentence in sentences:
            sentence_words = sentence.split()
            buffer_words = sum(len(s.split()) for s in buffer)
            if buffer_words + len(sentence_words) <= max_words:
                buffer.append(sentence)
            else:
                if buffer:
                    combined.append(' '.join(buffer))
                buffer = [sentence]
        if buffer:
            combined.append(' '.join(buffer))
        return combined