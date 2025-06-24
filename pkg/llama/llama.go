package llama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"strings"
	"sync"
	"time"
)

type ServerStatus struct {
	Message string
	IsError bool
}

type Server struct {
	llamaCmd *exec.Cmd
	port     string
	baseURL  string
	stderr   io.ReadCloser
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

func StartServer(llm string, port string) (*Server, error) {
	llamaCmd, err := exec.LookPath("llama-server")
	if err != nil {
		return nil, fmt.Errorf("llama-server not found: %w", err)
	}
	args := []string{"-hf", llm, "--port", port}
	cmd := exec.Command(llamaCmd, args...)

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stderr pipe: %w", err)
	}

	server := &Server{
		llamaCmd: cmd,
		port:     port,
		baseURL:  fmt.Sprintf("http://localhost:%s", port),
		stderr:   stderr,
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start llama-server: %w", err)
	}

	return server, nil
}

func (s *Server) StatusUpdates(ctx context.Context) <-chan ServerStatus {
	statusChan := make(chan ServerStatus, 10)

	go func() {
		var wg sync.WaitGroup
		defer func() {
			wg.Wait()
			close(statusChan)
		}()

		statusChan <- ServerStatus{Message: "Starting llama-server"}

		wg.Add(1)
		go func() {
			defer wg.Done()
			s.monitorErrors(statusChan)
		}()

		s.monitorHealth(ctx, statusChan)
		s.stderr.Close()
	}()
	return statusChan
}

func (s *Server) monitorErrors(statusChan chan<- ServerStatus) {
	scanner := bufio.NewScanner(s.stderr)
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case strings.Contains(line, "error: model is private or does not exist; if you are accessing a gated model, please provide a valid HF token"):
			statusChan <- ServerStatus{Message: "Model is private or does not exist; try using a different model", IsError: true}
		case strings.Contains(line, "error:"):
			statusChan <- ServerStatus{Message: line, IsError: true}
		}
	}
}

func (s *Server) monitorHealth(ctx context.Context, statusChan chan<- ServerStatus) {
	healthURL := s.baseURL + "/health"
	ticker := time.NewTicker(300 * time.Millisecond)
	defer ticker.Stop()

	timeout := time.After(30 * time.Second)

	for {
		select {
		case <-ctx.Done():
			return
		case <-timeout:
			statusChan <- ServerStatus{Message: "Server startup timed out", IsError: true}
		case <-ticker.C:
			resp, err := http.Get(healthURL)
			if err != nil {
				continue
			}
			resp.Body.Close()
			switch resp.StatusCode {
			case 200:
				statusChan <- ServerStatus{Message: "Server is ready", IsError: false}
				return
			case 503:
				statusChan <- ServerStatus{Message: "Loading model", IsError: false}
			default:
				statusChan <- ServerStatus{Message: fmt.Sprintf("Server not ready: %d", resp.StatusCode), IsError: true}
			}
		}
	}
}

func (s *Server) Stop() {
	if s.llamaCmd != nil && s.llamaCmd.Process != nil {
		s.llamaCmd.Process.Kill()
	}
}

func (s *Server) Inference(req InferenceReq) (<-chan InferenceResp, error) {

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
