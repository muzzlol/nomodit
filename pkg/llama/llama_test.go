package llama

import (
	"fmt"
	"testing"
	"time"
)

func TestLlamaServer(t *testing.T) {
	fmt.Println("Starting llama server")
	server, err := StartLlamaServer("unsloth/gemma-3-1b-it-GGUF", "8091")
	if err != nil {
		t.Fatalf("Failed to start llama server: %v", err)
	}
	defer server.Stop()

	fmt.Println("Server started")

	for {
		time.Sleep(1 * time.Second)
		fmt.Println("Server status:", server.status)

		if server.status == "Server is ready" {
			break
		}
	}

	inferenceReq1 := InferenceReq{
		Prompt: "Hello, how are you?",
		Temp:   0.7,
		Stream: true,
	}

	inferenceRespChan, err := server.Inference(inferenceReq1)
	if err != nil {
		t.Fatalf("Failed to get inference response: %v", err)
	}

	for resp := range inferenceRespChan {
		fmt.Println("Inference response:", resp.Content)
	}

	fmt.Println("Inference completed")
}
