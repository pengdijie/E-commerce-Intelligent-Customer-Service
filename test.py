import asyncio, sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
from memory.wiki_knowledge_base import WikiknowledgeBase

async def test():
    wiki = WikiknowledgeBase()
    
    questions = [
        '耳机右耳没声音怎么办？',
        '我想退款，需要多久到账？',
        '手环可以游泳用吗？',
    ]
    
    for q in questions:
        print(f'\n问：{q}')
        print('-' * 40)
        answer = await wiki.query(q, save_answer=True)
        print(answer[:300])
        print()

asyncio.run(test())