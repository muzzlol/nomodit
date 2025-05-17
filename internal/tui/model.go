package main

import (
	"fmt"
	"log"

	"github.com/charmbracelet/bubbles/spinner"
	// "github.com/charmbracelet/bubbles/cursor"
	// "github.com/charmbracelet/bubbles/textinput"

	"github.com/charmbracelet/bubbles/help"
	"github.com/charmbracelet/bubbles/key"

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

/**
	title1 = `
  ┳┓        ┓•
  ┃┃┏┓┏┳┓┏┓┏┫┓╋
  ┛┗┗┛┛┗┗┗┛┗┻┗┗

`
	title = `
	▖ ▖        ▌▘▗
	▛▖▌▛▌▛▛▌▛▌▛▌▌▜▘
	▌▝▌▙▌▌▌▌▙▌▙▌▌▐▖

`
*/

var (
	focusedStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("205")) // 205 - Hot Pink
	blurredStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("240")) // 240 - Gray
	cursorStyle  = focusedStyle
	accentStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("34"))         // 34 - Dark Cyan/Green
	textStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("252")).Render // 252 - Light Gray

	focusedButton = focusedStyle.Render("[ Submit ]")
	blurredButton = fmt.Sprintf("[ %s ]", blurredStyle.Render("Submit"))
)

func main() {
	p := tea.NewProgram(initialModel())
	if _, err := p.Run(); err != nil {
		log.Fatal(err)
	}

}

type model struct {
	title        string
	currentState state
	cursor       int
	submit       string
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
		key.WithHelp("tab/shift+tab", "navigate"),
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

	return model{
		title: accentStyle.Render(title),
		currentState: state{
			text:    textStyle(""),
			spinner: spinner.Pulse,
		},
		cursor: 1, // instructions part
		keys:   keys,
	}
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch {
		case key.Matches(msg, m.keys.Quit):
			return m, tea.Quit
		case key.Matches(msg, m.keys.Navigation):
			m.cursor++
		case key.Matches(msg, m.keys.Submit):
			m.cursor--
		}
	}

	return m, nil
}

func (m model) View() string {
	return m.title + gap + m.currentState.text + gap + m.submit + gap + m.help.View(m.keys)
}

func (m model) Init() tea.Cmd {
	return nil
}
