package tui

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/charmbracelet/bubbles/help"
	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/reflow/wordwrap"
	"github.com/sergi/go-diff/diffmatchpatch"

	"github.com/muzzlol/nomodit/pkg/llama"

	"github.com/spf13/viper"
)

const (
	gap   = "\n\n"
	title = `
  /|/_ _ _ _ _/._/_
 / /_/ / /_/_// /
	`
)

var (
	focusedButtonStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("255")). // White
				Background(lipgloss.Color("37"))   // Teal/Cyan

	blurredButtonStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("240")) // Gray

	focusedInputStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("51"))  // Lighter Teal/Cyan
	blurredInputStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("240")) // Gray
	cursorStyle       = focusedInputStyle
	accentStyle       = lipgloss.NewStyle().Foreground(lipgloss.Color("37"))                    // Teal/Cyan
	dangerStyle       = lipgloss.NewStyle().Foreground(lipgloss.Color("124"))                   // Red
	warningStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("208"))                   // Orange
	textStyle         = lipgloss.NewStyle().Foreground(lipgloss.Color("252"))                   // Light Gray
	deletedStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("9")).Strikethrough(true) // Bright Red
	addedStyle        = lipgloss.NewStyle().Foreground(lipgloss.Color("10"))                    // Bright Green

	submitFocusedButton = focusedButtonStyle.Render("[ Submit ]")
	submitBlurredButton = blurredButtonStyle.Render("[ Submit ]")
	copyFocusedButton   = focusedButtonStyle.Render("[ Copy ]")
	copyBlurredButton   = blurredButtonStyle.Render("[ Copy ]")
)

type model struct {
	server           *llama.Server
	serverReady      bool
	llm              string
	title            string
	currentState     state
	focusIndex       int
	focusables       []Focusable
	help             help.Model
	keys             keyMap
	suggestionsHelp  help.Model
	suggestionKeys   keyMap
	statusChan       <-chan llama.ServerStatus
	inferenceChan    <-chan llama.InferenceResp
	isInferring      bool
	inferenceBuilder strings.Builder
	output           viewport.Model
	width            int
	height           int
}

func diffing(og, new string) string {
	dmp := diffmatchpatch.New()
	diffs := dmp.DiffMain(og, new, false)
	s := wordwrap.NewWriter(98)
	for _, diff := range diffs {
		var text string
		switch diff.Type {
		case diffmatchpatch.DiffEqual:
			text = textStyle.Render(diff.Text)
		case diffmatchpatch.DiffInsert:
			text = addedStyle.Render(diff.Text)
		case diffmatchpatch.DiffDelete:
			text = deletedStyle.Render(diff.Text)
		}
		_, err := s.Write([]byte(text))
		if err != nil {
			log.Printf("error writing to wordwrap buffer: %v", err)
		}
	}
	_ = s.Close()
	return s.String()
}

func setupLogger() {
	// Log to a file for debugging purposes
	f, err := os.OpenFile("nomodit.log", os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0644)
	if err != nil {
		log.Fatalf("error opening file: %v", err)
	}
	log.SetOutput(f)
	log.Println("--- Log Start ---")
}

func Launch(instruction string) {
	setupLogger()
	m := InitialModel(instruction)
	defer func() {
		if m.server != nil {
			m.server.Stop()
		}
	}()

	p := tea.NewProgram(m)
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

	ta.FocusedStyle.Base = style.BorderForeground(lipgloss.Color("37")) // Teal/Cyan border
	ta.BlurredStyle.Base = style
	ta.Blur()

	return &fTextarea{Model: &ta}
}

type serverStatusMsg llama.ServerStatus
type serverReadyMsg struct{}
type inferenceMsg llama.InferenceResp
type inferenceDoneMsg struct{}

type keyMap struct {
	Navigation key.Binding
	Submit     key.Binding
	Quit       key.Binding
	Scroll     key.Binding
	Clear      key.Binding
}

func (k keyMap) ShortHelp() []key.Binding {
	return []key.Binding{k.Navigation, k.Submit, k.Quit, k.Scroll, k.Clear}
}

func (k keyMap) FullHelp() [][]key.Binding {
	return [][]key.Binding{
		{k.Navigation, k.Submit, k.Quit},
		{k.Scroll, k.Clear},
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
	Scroll: key.NewBinding(
		key.WithKeys("ctrl+u", "ctrl+d"),
		key.WithHelp("ctrl+u/d", "scroll up/down the output"),
	),
	Clear: key.NewBinding(
		key.WithKeys("ctrl+l"),
		key.WithHelp("ctrl+l", "clear input"),
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
	spinner spinner.Model
}

func InitialModel(instruction string) *model {
	// Create wrapper instances
	output := viewport.New(100, 20)
	output.Style = lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("240")).
		Width(100).
		Height(20)

	instructions := newFtextinput()
	instructions.Model.SetValue(instruction)
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
	input.Model.CharLimit = 0
	input.Model.SetWidth(100)
	input.Model.SetHeight(5)

	spinner := spinner.New(spinner.WithSpinner(spinner.Line), spinner.WithStyle(accentStyle))

	m := model{
		llm:         viper.GetString("llm"),
		serverReady: false,
		title:       accentStyle.Render(title),
		currentState: state{
			text:    accentStyle.Render("hi"),
			spinner: spinner,
		},
		focusIndex:      0,                                // instructions part
		focusables:      []Focusable{instructions, input}, // Use the wrappers
		keys:            keys,
		help:            help.New(),
		suggestionsHelp: help.New(),
		suggestionKeys:  suggestionKeys,
		isInferring:     false,
		output:          output,
		width:           100,
		height:          24,
	}
	return &m
}

func (m *model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case serverStatusMsg:
		if msg.IsError {
			m.currentState.text = dangerStyle.Render(msg.Message)
			return m, tea.Quit
		}
		m.currentState.text = accentStyle.Render(msg.Message)
		return m, m.checkServerStatus()
	case serverReadyMsg:
		m.serverReady = true
		m.currentState.text = accentStyle.Render("Ready! model: " + m.llm)
		return m, nil
	case inferenceMsg:
		log.Print(msg.Content)
		if msg.Stop {
			if msg.Content != "" {
				m.inferenceBuilder.WriteString(msg.Content)
			}
			ip := m.focusables[1].(*fTextarea)
			diff := diffing(ip.Model.Value(), m.inferenceBuilder.String())
			m.output.SetContent(diff)
			m.output.GotoBottom()
			return m, func() tea.Msg { return inferenceDoneMsg{} }
		}
		m.inferenceBuilder.WriteString(msg.Content)
		m.output.SetContent(m.inferenceBuilder.String())
		m.output.GotoBottom()
		return m, m.checkInference()
	case inferenceDoneMsg:
		m.isInferring = false
		m.currentState.text = accentStyle.Render("Done!")
		return m, nil
	case spinner.TickMsg:
		var cmd tea.Cmd
		m.currentState.spinner, cmd = m.currentState.spinner.Update(msg)
		return m, cmd
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		// Update viewport width to be responsive
		viewportWidth := min(100, m.width-4) // 4 for padding
		m.output.Width = viewportWidth
		// Update input width
		if len(m.focusables) > 1 {
			if ta, ok := m.focusables[1].(*fTextarea); ok {
				ta.Model.SetWidth(viewportWidth)
			}
		}
		return m, nil
	case tea.KeyMsg:
		switch {
		case key.Matches(msg, m.keys.Quit):
			m.server.Stop()
			return m, tea.Quit
		case key.Matches(msg, m.keys.Submit):
			if m.focusIndex == -1 {
				// TODO: copy functionality
			} else if m.focusIndex == len(m.focusables) && !m.isInferring {
				if !m.serverReady || m.isInferring {
					return m, nil
				}
				ip := m.focusables[1].(*fTextarea)

				instructions := m.focusables[0].(*fTextinput).Model.Value()
				if ip.Model.Value() == "" {
					m.currentState.text = warningStyle.Render("Enter text to submit")
					return m, nil
				}
				m.isInferring = true
				m.inferenceBuilder.Reset()
				m.output.SetContent("")
				m.currentState.text = accentStyle.Render("Generating")
				m.currentState.spinner = spinner.New(spinner.WithSpinner(spinner.Points), spinner.WithStyle(accentStyle))

				prompt := fmt.Sprintf("Instruction: %s\nText to fix: \"%s\"\n\nRespond with ONLY the fixed text, without any additional explanations, comments, or introductory phrases like \"Fixed text:\".", instructions, ip.Model.Value())

				req := llama.InferenceReq{
					Prompt:   prompt,
					Temp:     0.2,
					NPredict: 200,
				}

				var err error
				m.inferenceChan, err = m.server.Inference(req)
				if err != nil {
					m.currentState.text = dangerStyle.Render(err.Error())
					m.isInferring = false
					return m, nil
				}

				return m, tea.Batch(m.currentState.spinner.Tick, m.checkInference())
			}
		case key.Matches(msg, m.keys.Clear):
			if m.focusIndex == 0 {
				m.focusables[0].(*fTextinput).Model.Reset()
			} else {
				m.focusables[1].(*fTextarea).Model.Reset()
			}
		case key.Matches(msg, m.keys.Scroll):
			if msg.String() == "ctrl+u" {
				m.output.ScrollUp(2)
			} else {
				m.output.ScrollDown(2)
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

func (m *model) updateFocusables(msg tea.Msg) tea.Cmd {
	if m.focusIndex < 0 || m.focusIndex >= len(m.focusables) {
		return nil // No command to issue for components in focusables
	}
	var cmd tea.Cmd
	_, cmd = m.focusables[m.focusIndex].Update(msg)
	return cmd
}

func (m *model) checkInference() tea.Cmd {
	return func() tea.Msg {
		res, ok := <-m.inferenceChan
		if !ok {
			return inferenceDoneMsg{}
		}
		return inferenceMsg(res)
	}
}

func (m *model) checkServerStatus() tea.Cmd {
	return func() tea.Msg {
		status, ok := <-m.statusChan
		if !ok {
			return serverReadyMsg{}
		}
		return serverStatusMsg(status)
	}
}

func (m *model) Init() tea.Cmd {
	server, err := llama.StartServer(m.llm, "8091")
	if err != nil {
		m.currentState.text = dangerStyle.Render("Fatal: " + err.Error())
		return tea.Quit
	}
	m.server = server
	m.statusChan = m.server.StatusUpdates(context.Background())
	return tea.Batch(
		m.focusables[0].Focus(),
		m.currentState.spinner.Tick,
		m.checkServerStatus(),
	)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (m *model) View() string {
	var s strings.Builder

	// Calculate responsive width
	contentWidth := min(100, m.width-4) // Leave some margin
	if contentWidth < 50 {
		contentWidth = 50 // Minimum width
	}

	// Center the title
	centeredTitle := lipgloss.PlaceHorizontal(m.width, lipgloss.Center, m.title)
	s.WriteString(centeredTitle)
	s.WriteString("\n")

	// Center the status line
	statusText := m.currentState.text
	if !m.serverReady || m.isInferring {
		statusText += " " + m.currentState.spinner.View()
	}
	centeredStatus := lipgloss.PlaceHorizontal(m.width, lipgloss.Center, statusText)
	s.WriteString(centeredStatus)
	s.WriteString(gap)

	// Center the copy button
	copyButton := copyBlurredButton
	if m.focusIndex == -1 {
		copyButton = copyFocusedButton
	}
	centeredCopyButton := lipgloss.PlaceHorizontal(m.width, lipgloss.Center, copyButton)
	s.WriteString(centeredCopyButton)
	s.WriteString("\n")

	// Center the output viewport
	centeredOutput := lipgloss.PlaceHorizontal(m.width, lipgloss.Center, m.output.View())
	s.WriteString(centeredOutput)
	s.WriteString(gap)

	// Center the instructions input with border
	var borderStyle lipgloss.Style
	instructionWidth := min(98, contentWidth-2)
	if m.focusIndex == 0 {
		borderStyle = lipgloss.NewStyle().
			Width(instructionWidth).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("51")) // Lighter Teal/Cyan border when focused
	} else {
		borderStyle = lipgloss.NewStyle().
			Width(instructionWidth).
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("240")) // Gray border when blurred
	}
	instructionsWithBorder := borderStyle.Render(m.focusables[0].View())
	centeredInstructions := lipgloss.PlaceHorizontal(m.width, lipgloss.Center, instructionsWithBorder)
	s.WriteString(centeredInstructions)

	if m.focusIndex == 0 {
		s.WriteString("\n")
		centeredSuggestionHelp := lipgloss.PlaceHorizontal(m.width, lipgloss.Center, m.suggestionsHelp.View(m.suggestionKeys))
		s.WriteString(centeredSuggestionHelp)
	}
	s.WriteString("\n")

	// Center the input area
	centeredInput := lipgloss.PlaceHorizontal(m.width, lipgloss.Center, m.focusables[1].View())
	s.WriteString(centeredInput)
	s.WriteString("\n")

	// Center the submit button
	submitButton := submitBlurredButton
	if m.focusIndex == len(m.focusables) {
		submitButton = submitFocusedButton
	}
	centeredSubmit := lipgloss.PlaceHorizontal(m.width, lipgloss.Center, submitButton)
	s.WriteString(centeredSubmit)
	s.WriteString(gap)

	// Center the help
	centeredHelp := lipgloss.PlaceHorizontal(m.width, lipgloss.Center, m.help.View(m.keys))
	s.WriteString(centeredHelp)

	return s.String()
}
