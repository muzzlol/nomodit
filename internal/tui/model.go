package main

import (
	"log"
	"strings"

	"github.com/charmbracelet/bubbles/help"
	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/textinput"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/muzzlol/nomodit/pkg/locator"
)

const (
	gap   = "\n\n"
	title = `
  /|/_ _ _ _ _/._/_
 / /_/ / /_/_// /
	`
	llamaCLI = "llama-cli"
	llm      = "unsloth/gemma-3-1b-it-GGUF"
)

var (
	focusedButtonStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("255")). // White
				Background(lipgloss.Color("34"))   // Green

	blurredButtonStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("240")) // Gray

	focusedInputStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("118")) // Lighter Green
	blurredInputStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("240")) // Gray
	cursorStyle       = focusedInputStyle
	accentStyle       = lipgloss.NewStyle().Foreground(lipgloss.Color("34"))  // Green
	dangerStyle       = lipgloss.NewStyle().Foreground(lipgloss.Color("124")) // Red
	textStyle         = lipgloss.NewStyle().Foreground(lipgloss.Color("252")) // Light Gray

	submitFocusedButton = focusedButtonStyle.Render("[ Submit ]")
	submitBlurredButton = blurredButtonStyle.Render("[ Submit ]")
	copyFocusedButton   = focusedButtonStyle.Render("[ Copy ]")
	copyBlurredButton   = blurredButtonStyle.Render("[ Copy ]")
)

func main() {
	p := tea.NewProgram(initialModel())
	if _, err := p.Run(); err != nil {
		log.Fatal(err)
	}

}

type Focusable interface {
	Focus() tea.Cmd
	Blur()
	Update(msg tea.Msg) (tea.Model, tea.Cmd)
	View() string
}

type fTextinput struct {
	Model *textinput.Model
}

func (ft *fTextinput) Focus() tea.Cmd {
	ft.Model.PromptStyle = focusedInputStyle
	ft.Model.TextStyle = textStyle
	return ft.Model.Focus()
}
func (ft *fTextinput) Blur() {
	ft.Model.PromptStyle = blurredInputStyle
	ft.Model.TextStyle = blurredInputStyle
	ft.Model.Blur()
}
func (ft *fTextinput) View() string  { return ft.Model.View() }
func (ft *fTextinput) Init() tea.Cmd { return nil }
func (ft *fTextinput) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	updatedModel, cmd := ft.Model.Update(msg)
	ft.Model = &updatedModel
	return ft, cmd
}

type fTextarea struct {
	Model *textarea.Model
}

func (fta *fTextarea) Focus() tea.Cmd { return fta.Model.Focus() }
func (fta *fTextarea) Blur()          { fta.Model.Blur() }
func (fta *fTextarea) View() string   { return fta.Model.View() }
func (fta *fTextarea) Init() tea.Cmd  { return nil }
func (fta *fTextarea) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	updatedModel, cmd := fta.Model.Update(msg)
	fta.Model = &updatedModel
	return fta, cmd
}

func newFtextinput() *fTextinput {
	ti := textinput.New()
	ti.Cursor.Style = cursorStyle
	ti.CharLimit = 200

	ti.PromptStyle = focusedInputStyle
	ti.TextStyle = focusedInputStyle

	ti.Blur()

	return &fTextinput{Model: &ti}
}

func newFtextarea() *fTextarea {
	ta := textarea.New()
	ta.Cursor.Style = cursorStyle
	ta.ShowLineNumbers = false
	style := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("240")) // Gray border

	ta.FocusedStyle.Base = style.BorderForeground(lipgloss.Color("34"))
	ta.BlurredStyle.Base = style
	ta.Blur()

	return &fTextarea{Model: &ta}
}

type model struct {
	title           string
	currentState    state
	focusIndex      int
	focusables      []Focusable
	help            help.Model
	keys            keyMap
	suggestionsHelp help.Model
	suggestionKeys  keyMap
}

type keyMap struct {
	Navigation key.Binding
	Submit     key.Binding
	Quit       key.Binding
}

func (k keyMap) ShortHelp() []key.Binding {
	return []key.Binding{k.Navigation, k.Submit, k.Quit}
}

func (k keyMap) FullHelp() [][]key.Binding {
	return [][]key.Binding{
		{k.Navigation, k.Submit, k.Quit},
	}
}

var keys = keyMap{
	Navigation: key.NewBinding(
		key.WithKeys("tab", "shift+tab"),
		key.WithHelp("tab", "navigate"),
	),
	Submit: key.NewBinding(
		key.WithKeys("enter"),
		key.WithHelp("enter", "submit"),
	),
	Quit: key.NewBinding(
		key.WithKeys("ctrl+c", "esc"),
		key.WithHelp("ctrl+c/esc", "quit"),
	),
}

var suggestionKeys = keyMap{
	Navigation: key.NewBinding(
		key.WithKeys("up", "down"),
		key.WithHelp("Suggestions: ↑/↓", "navigate"),
	),
	Submit: key.NewBinding(
		key.WithKeys("enter"),
		key.WithHelp("enter", "accept"),
	),
}

type state struct {
	text    string
	spinner spinner.Spinner
}

func initialModel() model {
	// Create wrapper instances
	output := newFtextarea()
	output.Model.CharLimit = 500
	output.Model.SetWidth(100)
	output.Model.SetHeight(20)

	instructions := newFtextinput()
	instructions.Model.SetValue("Fix grammar")
	instructions.Model.SetSuggestions([]string{"Fix grammar and improve clarity of this text", "Fix grammar", "Fix grammar in this sentence", "Fix grammar in the sentence", "Fix grammar errors", "Fix grammatical errors", "Fix grammaticality", "Fix all grammatical errors", "Fix grammatical errors in this sentence", "Fix grammar errors in this sentence", "Fix grammatical mistakes in this sentence", "Fix grammaticality in this sentence", "Fix grammaticality of the sentence", "Fix disfluencies in the sentence", "Make the sentence grammatical", "Make the sentence fluent", "Fix errors in this text", "Update to remove grammar errors", "Remove all grammatical errors from this text", "Improve the grammar of this text", "Improve the grammaticality", "Improve the grammaticality of this text", "Improve the grammaticality of this sentence", "Grammar improvements", "Remove grammar mistakes", "Remove grammatical mistakes", "Fix the grammar mistakes", "Fix grammatical mistakes", "Clarify the sentence", "Clarify this sentence", "Clarify this text", "Write a clearer version for the sentence", "Write a clarified version of the sentence", "Write a readable version of the sentence", "Write a better readable version of the sentence", "Rewrite the sentence more clearly", "Rewrite this sentence clearly", "Rewrite this sentence for clarity", "Rewrite this sentence for readability", "Improve this sentence for readability", "Make this sentence better readable", "Make this sentence more readable", "Make this sentence readable", "Make the sentence clear", "Make the sentence clearer", "Clarify", "Make the text more understandable", "Make this easier to read", "Clarification", "Change to clearer wording", "Clarify this paragraph", "Use clearer wording", "Simplify the sentence", "Simplify this sentence", "Simplify this text", "Write a simpler version for the sentence", "Rewrite the sentence to be simpler", "Rewrite this sentence in a simpler manner", "Rewrite this sentence for simplicity", "Rewrite this with simpler wording", "Make the sentence simple", "Make the sentence simpler", "Make this text less complex", "Make this simpler", "Simplify", "Simplification", "Change to simpler wording", "Simplify this paragraph", "Simplify this text", "Use simpler wording", "Make this easier to understand", "Fix coherence", "Fix coherence in this sentence", "Fix coherence in the sentence", "Fix coherence in this text", "Fix coherence in the text", "Fix coherence errors", "Fix sentence flow", "Fix sentence transition", "Fix coherence errors in this sentence", "Fix coherence mistakes in this sentence", "Fix coherence in this sentence", "Fix coherence of the sentence", "Fix lack of coherence in the sentence", "Make the text more coherent", "Make the text coherent", "Make the text more cohesive", "Make the text more cohesive, logically linked and consistent as a whole", "Make the text more logical", "Make the text more consistent", "Improve the cohesiveness of the text", "Improve the consistency of the text", "Make the text clearer", "Improve the coherence of the text", "Formalize", "Improve formality", "Formalize the sentence", "Formalize this sentence", "Formalize the text", "Formalize this text", "Make this formal", "Make this more formal", "Make this sound more formal", "Make the sentence formal", "Make the sentence more formal", "Make the sentence sound more formal", "Write more formally", "Write less informally", "Rewrite more formally", "Write this more formally", "Rewrite this more formally", "Write in a formal manner", "Write in a more formal manner", "Rewrite in a more formal manner", "Remove POV", "Remove POVs", "Remove POV in this text", "Remove POVs in this text", "Neutralize this text", "Neutralize the text", "Neutralize this sentence", "Neutralize the sentence", "Make this more neutral", "Make this text more neutral", "Make this sentence more neutral", "Make this paragraph more neutral", "Remove unsourced opinions", "Remove unsourced opinions from this text", "Remove non-neutral POVs", "Remove non-neutral POV", "Remove non-neutral points of view", "Remove points of view", "Make this text less biased", "Paraphrase the sentence", "Paraphrase this sentence", "Paraphrase this text", "Paraphrase", "Write a paraphrase for the sentence", "Write a paraphrased version of the sentence", "Rewrite the sentence with different wording", "Use different wording", "Rewrite this sentence", "Reword this sentence", "Rephrase this sentence", "Rewrite this text", "Reword this text", "Rephrase this text"})
	instructions.Model.KeyMap.AcceptSuggestion = key.NewBinding(
		key.WithKeys("enter"),
		key.WithHelp("enter", "accept suggestion"),
	)
	instructions.Model.ShowSuggestions = true
	instructions.Model.Prompt = "Instructions: "
	instructions.Model.Focus()

	input := newFtextarea()
	input.Model.Placeholder = "Enter your text here"
	input.Model.CharLimit = 500
	input.Model.SetWidth(100)
	input.Model.SetHeight(5)

	m := model{
		title: accentStyle.Render(title),
		currentState: state{
			text:    textStyle.Render("helllllooo"),
			spinner: spinner.Pulse,
		},
		focusIndex:      1,                                        // instructions part
		focusables:      []Focusable{output, instructions, input}, // Use the wrappers
		keys:            keys,
		help:            help.New(),
		suggestionsHelp: help.New(),
		suggestionKeys:  suggestionKeys,
	}
	return m
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch {
		case key.Matches(msg, m.keys.Quit):
			return m, tea.Quit
		case key.Matches(msg, m.keys.Submit):
			if m.focusIndex == 0 {
				// copy funcitonality
			} else if m.focusIndex == len(m.focusables) {
				// submit functionality
			}
		case key.Matches(msg, m.keys.Navigation):
			if msg.String() == "tab" {
				m.focusIndex++
			} else {
				m.focusIndex--
			}
			if m.focusIndex < -1 {
				m.focusIndex = len(m.focusables)
			} else if m.focusIndex > len(m.focusables) {
				m.focusIndex = -1
			}
			cmds := make([]tea.Cmd, len(m.focusables))
			for i := 0; i <= len(m.focusables)-1; i++ {
				if i == m.focusIndex {
					// Set focused state
					cmds[i] = m.focusables[i].Focus()
					continue
				}
				// Remove focused state
				m.focusables[i].Blur()
			}
			return m, tea.Batch(cmds...)
		}
	}
	// Handle character input and blinking
	cmd := m.updateFocusables(msg)
	return m, cmd
}

func (m model) updateFocusables(msg tea.Msg) tea.Cmd {
	if m.focusIndex < 0 || m.focusIndex >= len(m.focusables) {
		return nil // No command to issue for components in focusables
	}
	var cmd tea.Cmd
	_, cmd = m.focusables[m.focusIndex].Update(msg)
	return cmd
}

func (m model) View() string {
	var s strings.Builder
	s.WriteString(m.title)
	s.WriteString("\n")
	// add spinner
	if m.currentState.text != "" {
		s.WriteString(m.currentState.text)
	}
	s.WriteString(gap)
	if m.focusIndex == -1 {
		s.WriteString(copyFocusedButton)
	} else {
		s.WriteString(copyBlurredButton)
	}
	s.WriteString("\n")

	s.WriteString(m.focusables[0].View()) // output area
	s.WriteString(gap)

	var borderStyle lipgloss.Style
	if m.focusIndex == 1 {
		borderStyle = lipgloss.NewStyle().
			Width(98).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("118")) // Lighter Green border when focused
	} else {
		borderStyle = lipgloss.NewStyle().
			Width(98).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("240")) // Gray border when blurred
	}
	s.WriteString(borderStyle.Render(m.focusables[1].View())) // instructions
	if m.focusIndex == 1 {
		s.WriteString("\n")
		s.WriteString(m.suggestionsHelp.View(m.suggestionKeys))
	}
	s.WriteString("\n")
	s.WriteString(m.focusables[2].View()) // input area
	s.WriteString("\n")
	// Submit Button
	if m.focusIndex == len(m.focusables) {
		s.WriteString(submitFocusedButton)
	} else {
		s.WriteString(submitBlurredButton)
	}
	s.WriteString(gap)
	// Help
	s.WriteString(m.help.View(m.keys))

	return s.String()
}

func (m model) Init() tea.Cmd {
	_, err := locator.CheckLlama()
	if err != nil {
		m.currentState.text = dangerStyle.Render("llama-cli not found")
	}
	return m.focusables[1].Focus()
}
