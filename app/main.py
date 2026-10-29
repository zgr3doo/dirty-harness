import asyncio
import logging
from pathlib import Path
from datetime import datetime

from llm_provider import connect_to_llm
from tools import read_file, write_file, create_file, ls, glob, run_pytest, validate_puml, validate_python_file, analyze_repository
from input import system_prompt, prompt
from log_parser import LogParser
from agent import Agent

def setup_loggers():
    Path("../logs/events").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # Logger 1
    event_logger = logging.getLogger("deep")
    event_logger.setLevel(logging.INFO)

    event_handler = logging.FileHandler(
        f"/logs/events/{timestamp}.log",
        encoding="utf-8"
    )
    event_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    event_logger.addHandler(event_handler)

    return event_logger

async def main():
    logger = setup_loggers()

    abot = Agent(connect_to_llm(),
                 [analyze_repository, read_file, write_file, create_file, ls, glob, run_pytest, validate_puml, validate_python_file],
                 system_prompt=system_prompt.messages[0].content)

    new_parser = LogParser()

    async for event in abot.graph.astream(
            {
                "messages": prompt.messages
            },
            config={"configurable": {"thread_id": "1"}},
            stream_mode=["updates"],
            subgraphs=True,
    ):
        logger.info(event)

        for n_event in new_parser.parse_chunk(event):
            logger.info(n_event)
            if n_event.event_type in ("human_message", "model_response"):
                print(f"{n_event.description} \n{"-" * 80}\nTokeny: {n_event.total_tokens} \n")

if __name__ == "__main__":
    start = datetime.now()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPrzerwano program (Ctrl+C).")
    finally:
        end = datetime.now()
        print("*" * 150)
        print(f"Czas trwania: {end - start}")