/*
Copyright © 2024 Muzz Khan muzxmmilkhxn@gmail.com
*/
package cmd

import (
	"fmt"
	"os"

	"github.com/charmbracelet/lipgloss"
	"github.com/muzzlol/nomodit/internal/tui"
	"github.com/muzzlol/nomodit/pkg/config"
	"github.com/muzzlol/nomodit/pkg/llama"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	LLM         string
	Instruction string
	dangerStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("124"))
)

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "cli := nomodit [flags] [\"text\"] \ntui := nomodit [flags]",
	Short: "Nomodit is a CLI/TUI for inferencing LLMs for language tasks",
	Long: `Nomodit is a CLI/TUI for inferencing LLMs for language tasks.
It allows you to use the nomodit series of models ( more about it here: https://github.com/muzzlol/nomodit ) and also any other model that supports the GGUF format.
	`,
	Args: cobra.MinimumNArgs(1),
	PreRun: func(cmd *cobra.Command, args []string) {
		if err := config.Load(); err != nil {
			fmt.Printf("failed to load config: %v, \nusing default values: llm: %v, instruction: %v\n", err, LLM, Instruction)
		}
		if cmd.Flags().Changed("llm") {
			viper.Set("llm", LLM)
			if err := config.Save(); err != nil {
				fmt.Printf("failed to save config: %v\n", err)
			}
		}
	},
	Run: func(cmd *cobra.Command, args []string) {
		if len(args) == 0 {
			tui.Launch()
		} else {
			if args[0] == "" {
				cmd.PrintErrln(dangerStyle.Render("Please provide some text to edit."))
				return
			}
			prompt := args[0]
			server, err := llama.StartLlamaServer(LLM, "8091")
			if err != nil {
				cmd.PrintErrln(dangerStyle.Render(err.Error()))
				return
			}
			defer server.Stop()

			inferenceReq := llama.InferenceReq{
				Prompt: prompt,
				Temp:   0.3,
				Stream: true,
			}
			respStream, err := server.Inference(inferenceReq)
			if err != nil {
				cmd.PrintErrln(dangerStyle.Render(err.Error()))
				return
			}

			for resp := range respStream {
				fmt.Print(resp.Content)
			}

			fmt.Println("\n\n*Inference completed*")

		}
	},
}

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() {
	err := rootCmd.Execute()
	if err != nil {
		os.Exit(1)
	}
}

func init() {
	rootCmd.Flags().StringVarP(&LLM, "llm", "m", "unsloth/gemma-3-1b-it-GGUF", "LLM to be used")
	rootCmd.Flags().StringVarP(&Instruction, "instruction", "i", "Fix grammar and improve clarity of this text", "Instructions to use for the LLM")

	viper.BindPFlag("llm", rootCmd.Flags().Lookup("llm"))

}
