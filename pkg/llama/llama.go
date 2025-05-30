package llama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"strings"
	"time"
)

type LlamaServer struct {
	llamaCmd *exec.Cmd
	port     string
	baseURL  string
	status   string
}

type InferenceReq struct {
	Prompt      string  `json:"prompt"`
	Temp        float32 `json:"temp,omitempty"`
	Stream      bool    `json:"stream,omitempty"`
	NPredict    int     `json:"n_predict,omitempty"`
	CachePrompt bool    `json:"cache_prompt"`
}

type InferenceResp struct {
	Content string `json:"content"`
	Stop    bool   `json:"stop"`
}

func StartLlamaServer(llm string, port string) (*LlamaServer, error) {
	llamaCmd, err := exec.LookPath("llama-server")
	if err != nil {
		return nil, fmt.Errorf("llama-server not found: %w", err)
	}
	args := []string{"-hf", llm, "--port", port}
	cmd := exec.Command(llamaCmd, args...)

	server := &LlamaServer{
		llamaCmd: cmd,
		port:     port,
		baseURL:  fmt.Sprintf("http://localhost:%s", port),
		status:   "starting server",
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start llama-server: %w", err)
	}

	if err := server.waitForServer(30 * time.Second); err != nil {
		server.Stop()
		return nil, fmt.Errorf("failed to start within timeout: %w", err)
	}

	return server, nil
}

func (s *LlamaServer) waitForServer(timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			resp, err := http.Get(s.baseURL + "/health")
			if err == nil {
				resp.Body.Close()
				if resp.StatusCode == 503 {
					s.status = "Loading model"
				} else if resp.StatusCode == 200 {
					s.status = "Server is ready"
					return nil
				}
			}
		}
	}
}

func (s *LlamaServer) Stop() {
	if s.llamaCmd != nil && s.llamaCmd.Process != nil {
		s.llamaCmd.Process.Kill()
	}
}

func (s *LlamaServer) Inference(req InferenceReq) (<-chan InferenceResp, error) {

	req.Stream = true
	req.CachePrompt = false

	jsonReq, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	client := &http.Client{
		Timeout: 3 * time.Minute,
	}

	resp, err := client.Post(
		s.baseURL+"/completion",
		"application/json",
		bytes.NewBuffer(jsonReq),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	if resp.StatusCode != 200 {
		resp.Body.Close()
		return nil, fmt.Errorf("server returned status %d", resp.StatusCode)
	}

	respChan := make(chan InferenceResp, 100)

	go func() {
		defer resp.Body.Close()
		defer close(respChan)

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()

			if line == "" || !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")

			var inferenceResp InferenceResp // new struct on every req so as to allow handling of data before replacing
			if err := json.Unmarshal([]byte(data), &inferenceResp); err != nil {
				continue
			}

			respChan <- inferenceResp

			if inferenceResp.Stop {
				break
			}
		}
	}()

	return respChan, nil
}
