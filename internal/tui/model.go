package main

import (
	"fmt"
	"log"
	"strings"

	"github.com/charmbracelet/bubbles/help"
	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
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

type Focusable interface {
	Focus() tea.Cmd
	Blur()
	Update(msg tea.Msg) (tea.Model, tea.Cmd)
	View() string
}

type fTextinput struct {
	Model *textinput.Model
}

func (ft *fTextinput) Focus() tea.Cmd { return ft.Model.Focus() }
func (ft *fTextinput) Blur()          { ft.Model.Blur() }
func (ft *fTextinput) View() string   { return ft.Model.View() }
func (ft *fTextinput) Init() tea.Cmd  { return nil }
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
	ti.Width = 100
	ti.Cursor.Style = cursorStyle
	ti.CharLimit = 200

	return &fTextinput{Model: &ti}
}

func newFtextarea() *fTextarea {
	ta := textarea.New()
	ta.Cursor.Style = cursorStyle
	ta.ShowLineNumbers = false
	return &fTextarea{Model: &ta}
}

type model struct {
	title        string
	currentState state
	focusIndex   int
	focusables   []Focusable
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
	// Create wrapper instances
	output := newFtextarea()
	output.Model.CharLimit = 500
	output.Model.SetWidth(100)
	output.Model.SetHeight(20)

	instructions := newFtextinput()
	instructions.Model.Placeholder = "Enter your instructions here (default: fix grammatical errors)"
	instructions.Model.Prompt = "Instructions: "
	instructions.Model.Focus()

	input := newFtextarea()
	input.Model.CharLimit = 500
	input.Model.SetWidth(100)
	input.Model.SetHeight(5)

	m := model{
		title: accentStyle.Render(title),
		currentState: state{
			text:    textStyle(""),
			spinner: spinner.Pulse,
		},
		focusIndex: 1,                                        // instructions part
		focusables: []Focusable{output, instructions, input}, // Use the wrappers
		keys:       keys,
		help:       help.New(),
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
	// only updates currently focused
	cmds := make([]tea.Cmd, len(m.focusables))
	for i := range m.focusables {
		_, cmds[i] = m.focusables[i].Update(msg) // ignoring tea.Model cause wrappers update internally
	}
	return tea.Batch(cmds...)
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
	s.WriteString(m.focusables[0].View()) // view
	// reasoning traces view
	s.WriteString(gap)
	s.WriteString(m.focusables[1].View()) // instructions
	s.WriteString("\n")
	s.WriteString(m.focusables[2].View()) // input
	// submit button
	if m.focusIndex == len(m.focusables) {
		s.WriteString(submitFocusedButton)
	} else {
		s.WriteString(submitBlurredButton)
	}
	s.WriteString(gap)
	s.WriteString(m.help.View(m.keys)) // help

	return s.String()
}

func (m model) Init() tea.Cmd {
	return m.focusables[1].Focus()
}
