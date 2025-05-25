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
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

type Config struct {
	LLM         string
	Instruction string
}

var (
	cfg         Config
	dangerStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("124"))
)

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "nomodit",
	Short: "Nomodit is a cli/tui to inference LLMs for language tasks",
	Long:  ``,
	Args: func(cmd *cobra.Command, args []string) error {
		if len(args) > 1 {
			return fmt.Errorf("please wrap your text in quotes")
		}
		return nil
	},
	PreRun: func(cmd *cobra.Command, args []string) {
		if err := config.Load(); err != nil {
			fmt.Printf("failed to load config: %v, \nusing default values: %v\n", err, cfg)
		}
		if cmd.Flags().Changed("llm") {
			viper.Set("llm", cfg.LLM)
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
			text := args[0]
			fmt.Println("text", text)
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
	rootCmd.Flags().StringVarP(&cfg.LLM, "llm", "l", "unsloth/gemma-3-1b-it-GGUF", "LLM model to use")
	rootCmd.Flags().StringVarP(&cfg.Instruction, "instruction", "i", "Fix grammar and improve clarity of this text", "Instructions to use for the LLM")

	viper.BindPFlag("llm", rootCmd.Flags().Lookup("llm"))

}
