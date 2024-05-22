.PHONY: start

livekit-server:
	nohup livekit-server --dev &

agent: livekit-server
	python agent.py start

.PHONY: format
format:
	black .
	isort .
