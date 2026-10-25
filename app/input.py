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

Jesteś głównym agentem.
Nie wykonuj zadań związanych z analizą plików, jeżeli użytkownik pyta o analize zawartości plików lub potrzebujesz
przeanalizować większą liczbę plików, deleguj to do file_reader.
Następnie wykorzystaj jego podsumowanie do udzielenia odpowiedzi.
"""




done = 'istnieje /workspace/results/result.md'

question = """
ile to 2 + 2 ?
zapisz odpowiedz w /workspace/results/result.md
"""


# done = 'istnieje r5.md'

# done = 'aplikacja dziala'

# done = 'testy przechodza w /home/zgr3doo/DEV/z3/workspace/full-blast/test/test_group/game_manager_test.gd'

# done = '1 etap planu z /home/zgr3doo/DEV/z3/workspace/plan8.md zostal zaimplementowany'

# done = 'plan /home/zgr3doo/DEV/z3/workspace/plan6.md zostal zaimplementowany i testy przechodza'
context = """"""
# context = """
# nie mozna sprawdzic czy testy przechodza w projekcie full-blast
# opis problemu znajduje sie w /home/zgr3doo/DEV/z3/workspace/plan5.md
# """
# question = """
# zaimplementuj 1 etap planu poprawy run_gdunit_tests tak aby mozna uzywac narzedzia do odplania wszystkich testow w full-blast
# plan znajduje sie w /home/zgr3doo/DEV/z3/workspace/plan8.md
# """

# question = """
# przeanalizuj plan poprawy run_gdunit_tests tak aby mozna uzywac narzedzia do odplania wszystkich testow w full-blast
# plan znajduje sie w /home/zgr3doo/DEV/z3/workspace/plan7.md
# podziel 1 krok planu na czesci na 3 etapy "auto", "headless", "xvfb" tak aby mozna je wykonac w sekwencji i zapisz plan do /home/zgr3doo/DEV/z3/workspace/plan8.md
# """

# question = """
# zrob plan poprawy run_gdunit_tests tak aby stalo sie to mozliwe
# zapisz plan do  /home/zgr3doo/DEV/z3/workspace/plan6.md
# """

# question = """
# napisz aplikacje w https://textual.textualize.io/
# powinna dzialac jak wrapper dla aplikacji do inferencji z modelem ai
# powinna byc w stanie wyswietlac odpowiedzi modelu i akceptowac input z pola tekstowego
# """

# question = """
# stworz plan naprawy problemu wykrytego w analizie r2.md
# aser wystrzelony tuż przed końcem wleci na Level2 (a po ostatnim poziomie zostanie nad MainMenu) — fix: 1 linia (`add_child` do sceny zamiast `/root`)
# opisz krok po kroku co nalezy poprawic
# zapisz rezultat do r5.md
# """

# question = """
# przeanalizuj test test_start_game_sets_state w /app/workspace/full-blast/test/test_group/game_manager_test.gd
# i opisz dokladnie krok po kroku co ten test robi rezultat zapisz w /test-summary-1.md
# """

# question = """
# sprawdz czy testy przechodza w projekcie /home/zgr3doo/DEV/z3/workspace/full-blast
# zbadaj to i opisz rezultat w  /test-results.md
# """

# question = """
# uzyj mcp project_list do ustalenia polozenia projektu /home/zgr3doo/DEV/z3/workspace/full-blast
# i zapisz rezultat do /results.md
# """

# question = """
# odpal test /full-blast/test/test_group/player_test.gd
# i zapisz rezultat do /results.md
# """

# question = """
# sprawdz jakie masz dostepne narzedzia mcp wypisz je do /mcp-list.md
# """

# question = """
# Podczas startu pojawia mcp sie blad
#
# ⠙ Ładuje MCP toolsTraceback (most recent call last):
#   File "/home/zgr3doo/DEV/z3/workspace/g-mcp/godot_mcp/server.py", line 41, in <module>
#     from godot_mcp.core import (
#     ...<8 lines>...
#     )
# ModuleNotFoundError: No module named 'godot_mcp'
#
# zbadaj to i opisz rezultat w  /home/zgr3doo/DEV/z3/workspace/plan2.md
# """

# question = """
# Podczas startu pojawia mcp sie blad
#
# ⠙ Ładuje MCP toolsTraceback (most recent call last):
#   File "/home/zgr3doo/DEV/z3/workspace/g-mcp/godot_mcp/server.py", line 41, in <module>
#     from godot_mcp.core import (
#     ...<8 lines>...
#     )
# ModuleNotFoundError: No module named 'godot_mcp'
#
# zbadaj to i opisz rezultat w  /home/zgr3doo/DEV/z3/workspace/plan2.md
# """

system_prompt = system_template.invoke({
    'system': system,
    'done': done
})

prompt = template.invoke({
    "context": context,
    "question": question
})

