/*
Copyright © 2024 Muzz Khan muzxmmilkhxn@gmail.com
*/
package cmd

import (
	"os"

	"github.com/spf13/cobra"
)

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "nomodit",
	Short: "A CLI tool for performant text enhancement w/ punctuation correction and synonym suggestions",
	Long: `This CLI tool empowers users to edit and enhance their text by providing punctuation 
corrections and synonym suggestions. It prioritizes user control by allowing interactive 
synonym selection for non-stop words, enabling the creation of customized and polished text.

Features:
  • Punctuation Correction: Automatically adjusts punctuation errors in user-input text
    for clarity and correctness
  • Synonym Suggestions: Highlights non-stop words and provides interactive synonym
    selection
  • Editable Output: Outputs the corrected and customized text ready for use

The tool combines Go's performance with Python's NLP capabilities to provide an efficient
and user-friendly text enhancement experience.`,
	// Uncomment the following line if your bare application
	// has an action associated with it:
	// Run: func(cmd *cobra.Command, args []string) { },
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
	// Here you will define your flags and configuration settings.
	// Cobra supports persistent flags, which, if defined here,
	// will be global for your application.

	// rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.nomodit.yaml)")

	// Cobra also supports local flags, which will only run
	// when this action is called directly.
	rootCmd.Flags().BoolP("toggle", "t", false, "Help message for toggle")
}
