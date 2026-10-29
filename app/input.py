from langchain_core.prompts import ChatPromptTemplate

system_template = ChatPromptTemplate.from_messages([('system', '{system} \n DONE = {done}')])
template = ChatPromptTemplate.from_messages([('human', 'Kontekst: {context} \n Pytanie: {question}')])


system = """
Jesteś kodujacym agentem nastawionym na realizację zadań.
W przypadku złożonych zadań składających się z wielu etapów:
najpierw utwórz listę zadań (TODO),
oznacz zadania jako in_progress, gdy nad nimi pracujesz,
oznacz zadania jako completed, gdy je ukończysz,
aktualizuj listę zadań, gdy plan ulegnie zmianie.
"""


context = ""

done = 'istnieje /results.md'

question = """
ile to 2 + 2 ?
zapisz odpowiedz w /results.md
"""


system_prompt = system_template.invoke({
    'system': system,
    'done': done
})

prompt = template.invoke({
    "context": context,
    "question": question
})

