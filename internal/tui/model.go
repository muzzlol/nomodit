package main

import (
	"fmt"
	"log"
	"strings"

	"github.com/charmbracelet/bubbles/help"
	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

const (
	gap   = "\n\n"
	title = `
  /|/_ _ _ _ _/._/_
 / /_/ / /_/_// /
	`
)

var (
	focusedStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("205")) // 205 - Hot Pink
	blurredStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("240")) // 240 - Gray
	cursorStyle  = focusedStyle
	accentStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("34"))         // 34 - Dark Cyan/Green
	textStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("252")).Render // 252 - Light Gray
	noStyle      = lipgloss.NewStyle()

	submitFocusedButton = focusButton("Submit")
	submitBlurredButton = blurButton("Submit")
	copyFocusedButton   = focusButton("Copy")
	copyBlurredButton   = blurButton("Copy")
)

func focusButton(str string) string {
	return focusedStyle.Render(fmt.Sprintf("[ %s ]", str))
}

func blurButton(str string) string {
	return blurredStyle.Render(fmt.Sprintf("[ %s ]", str))
}

func main() {
	p := tea.NewProgram(initialModel())
	if _, err := p.Run(); err != nil {
		log.Fatal(err)
	}

}

type model struct {
	title        string
	currentState state
	focusIndex   int
	inputs       []textinput.Model
	help         help.Model
	keys         keyMap
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

type state struct {
	text    string
	spinner spinner.Spinner
}

func initialModel() model {
	inputs := make([]textinput.Model, 3) // 0: view, 1: instructions, 2: input

	for i := range inputs {
		switch i {
		case 0:
			// view
		case 1:
			instructions := textinput.New()
			instructions.Placeholder = "Enter your instructions here (default: fix grammatical errors)"
			instructions.CharLimit = 200
			instructions.Width = 150
			// instructions.SetSuggestions([]string{"fix grammatical errors", "fix spelling errors", "fix punctuation errors"})
			// instructions.ShowSuggestions = true
			instructions.Cursor.Style = cursorStyle
			inputs[1] = instructions
		case 2:
			input := textinput.New()
			input.Placeholder = "Enter your input here"
			input.CharLimit = 200
			input.Width = 40
			input.Cursor.Style = cursorStyle
			inputs[2] = input
		}
	}

	return model{
		title: accentStyle.Render(title),
		currentState: state{
			text:    textStyle(""),
			spinner: spinner.Pulse,
		},
		focusIndex: 1, // instructions part
		inputs:     inputs,
		keys:       keys,
		help:       help.New(),
	}
}

func (m *model) updateInputs(msg tea.Msg) tea.Cmd {
	cmds := make([]tea.Cmd, len(m.inputs))

	// Only text inputs with Focus() set will respond, so it's safe to simply
	// update all of them here without any further logic.
	for i := range m.inputs {
		m.inputs[i], cmds[i] = m.inputs[i].Update(msg)
	}

	return tea.Batch(cmds...)
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
			} else if m.focusIndex == len(m.inputs) {
				// submit functionality
			}
		case key.Matches(msg, m.keys.Navigation):
			if msg.String() == "tab" {
				m.focusIndex++
			} else {
				m.focusIndex--
			}
			if m.focusIndex < -1 {
				m.focusIndex = len(m.inputs)
			} else if m.focusIndex > len(m.inputs) {
				m.focusIndex = -1
			}
			cmds := make([]tea.Cmd, len(m.inputs))
			for i := 0; i <= len(m.inputs)-1; i++ {
				if i == m.focusIndex {
					// Set focused state
					cmds[i] = m.inputs[i].Focus()
					m.inputs[i].PromptStyle = focusedStyle
					m.inputs[i].TextStyle = focusedStyle
					continue
				}
				// Remove focused state
				m.inputs[i].Blur()
				m.inputs[i].PromptStyle = noStyle
				m.inputs[i].TextStyle = noStyle
			}
			return m, tea.Batch(cmds...)
		}
	}
	// Handle character input and blinking
	cmd := m.updateInputs(msg)
	return m, cmd
}

func (m model) View() string {
	var s strings.Builder
	s.WriteString(m.title)
	s.WriteString(gap)
	// spinner
	s.WriteString(m.currentState.text)
	s.WriteString(gap)
	// copy button
	if m.focusIndex == -1 {
		s.WriteString(copyFocusedButton)
	} else {
		s.WriteString(copyBlurredButton)
	}
	s.WriteString(m.inputs[0].View()) // view
	// reasoning traces view
	s.WriteString(gap)
	s.WriteString(m.inputs[1].View()) // instructions
	s.WriteString("\n")
	s.WriteString(m.inputs[2].View()) // input
	// submit button
	if m.focusIndex == len(m.inputs) {
		s.WriteString(submitFocusedButton)
	} else {
		s.WriteString(submitBlurredButton)
	}
	s.WriteString(gap)
	s.WriteString(m.help.View(m.keys)) // help

	return s.String()
}

func (m model) Init() tea.Cmd {
	return textinput.Blink
}
