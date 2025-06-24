package llama

import (
	"context"
	"fmt"
	"testing"
)

func TestLlamaServer(t *testing.T) {
	fmt.Println("Starting llama server")
	server, err := StartServer("unsloth/gemma-3-1b-it-GGUF", "8091")
	if err != nil {
		t.Fatalf("Failed to start llama server: %v", err)
	}
	t.Cleanup(server.Stop)

	statusChan := server.StatusUpdates(context.Background())

	checkStatus(statusChan)

	inferenceReq1 := InferenceReq{
		Prompt:   "what is the capital of france?",
		Temp:     0.1,
		NPredict: 100,
	}

	inferenceRespChan, err := server.Inference(inferenceReq1)
	if err != nil {
		t.Fatalf("Failed to get inference response: %v", err)
	}

	for resp := range inferenceRespChan {
		fmt.Println("response 1:", resp.Content)
	}
	fmt.Println("starting second inference")

	inferenceReq2 := InferenceReq{
		Prompt:   "What did i ask you before this?",
		Temp:     0.1,
		NPredict: 100,
	}

	inferenceRespChan, err = server.Inference(inferenceReq2)
	if err != nil {
		t.Fatalf("Failed to get inference response: %v", err)
	}

	for resp := range inferenceRespChan {
		fmt.Println("\n\nresponse 2:", resp.Content)
	}
	fmt.Println("Inference completed")
}

func checkStatus(statusChan <-chan ServerStatus) {
	for status := range statusChan {
		fmt.Println("Server status:", status.Message)
		if status.IsError {
			fmt.Printf("Server error: %s", status.Message)
		}
	}
}
