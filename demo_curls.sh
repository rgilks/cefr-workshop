#!/bin/bash
# Demo curl commands for testing CEFR scoring at different levels

echo "=== A1 Level (very simple, fragmented) ==="
curl -s -X POST "https://rob-gilks--cefr-api-cefrservice-serve.modal.run/score" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am student. I like school. My teacher is nice. I have friend. We play."}' | jq .

echo -e "\n=== A2 Level (simple sentences, basic connectors) ==="
curl -s -X POST "https://rob-gilks--cefr-api-cefrservice-serve.modal.run/score" \
  -H "Content-Type: application/json" \
  -d '{"text": "I like to go shopping with my friends. We go to the mall every weekend. I usually buy clothes and sometimes we eat pizza. It is very fun."}' | jq .

echo -e "\n=== B1 Level (connected, clear, personal topics) ==="
curl -s -X POST "https://rob-gilks--cefr-api-cefrservice-serve.modal.run/score" \
  -H "Content-Type: application/json" \
  -d '{"text": "Last weekend I visited my grandmother. She lives in a small village near the mountains. We cooked dinner together and talked about old times. I really enjoyed spending time with her."}' | jq .

echo -e "\n=== B2 Level (clear, detailed, can argue a point) ==="
curl -s -X POST "https://rob-gilks--cefr-api-cefrservice-serve.modal.run/score" \
  -H "Content-Type: application/json" \
  -d '{"text": "The question of whether social media has had a positive or negative impact on society is complex. On one hand, platforms like Facebook have enabled unprecedented global connectivity. On the other hand, there is growing evidence that excessive use may contribute to mental health issues, particularly among young people."}' | jq .

echo -e "\n=== C1/C2 Level (sophisticated, nuanced) ==="
curl -s -X POST "https://rob-gilks--cefr-api-cefrservice-serve.modal.run/score" \
  -H "Content-Type: application/json" \
  -d '{"text": "The proliferation of misinformation in digital spaces represents a fundamental challenge to democratic discourse. While technological solutions such as fact-checking algorithms offer some mitigation, the underlying issue stems from deeper epistemological fragmentation in contemporary society."}' | jq .
