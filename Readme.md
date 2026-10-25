1) Setup - build image

```
uv sync
docker compose build --no-cache --pull
```

2) add llm provider

go to app/llm_provider.py and enter your ip
```
LLM_PROVIDER_IP="*.*.*.*:*"
```

3) edit your request

go to app/input.py and edit question and done
```
done = "..." <-- condition of done

question = "..." <-- task
```


4) run container

```
docker compose run --rm deepagent
```