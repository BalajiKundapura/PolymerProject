from testExtractor.txtNormalize import preprocess_file
from testExtractor.sentenceDefiner import extract_conditions
from polymerSubject import extract_all_polymers

def run(file_path: str):
    sentences = preprocess_file(file_path)
    print('File:', file_path)
    print('Sentences:', len(sentences))
    print('First 3 sentences:')
    for s in sentences[:3]:
        print('-', s)
    events = extract_conditions(sentences, extract_all_polymers)
    print('Events:', len(events))
    if events:
        print('First event:')
        print(events[0])

if __name__ == '__main__':
    run('rawData/paper.txt')
