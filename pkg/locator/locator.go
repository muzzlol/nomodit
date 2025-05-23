package locator

import (
	"fmt"
	"os/exec"
	"runtime"
)

func FindOs() (string, error) {
	os := runtime.GOOS
	switch os {
	case "darwin", "linux", "freebsd", "openbsd", "netbsd", "dragonfly", "solaris", "aix":
		return "unix", nil
	case "windows":
		return "windows", nil
	default:
		return "", fmt.Errorf("unsupported OS: %s", os)
	}
}

func CheckLlama() (string, error) {
	_, err := exec.LookPath("llama-cli")
	if err != nil {
		return "", fmt.Errorf("llama-cli not found: %w", err)
	}
	return "llama-cli", nil
}
