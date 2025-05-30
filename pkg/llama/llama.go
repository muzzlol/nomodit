package llama

import (
	"context"
	"fmt"
	"net/http"
	"os/exec"
	"time"
)

type LlamaServer struct {
	llamaCmd *exec.Cmd
	port     string
	baseURL  string
	status   string
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
