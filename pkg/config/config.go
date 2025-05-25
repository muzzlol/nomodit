package config

import (
	"os"
	"path/filepath"

	"github.com/spf13/viper"
)

func Load() error {
	viper.SetConfigName("config")
	viper.SetConfigType("env")

	viper.AddConfigPath(".")
	viper.AddConfigPath("./pkg/config")
	home, err := os.UserHomeDir()
	if err == nil {
		viper.AddConfigPath(home + "/.nomodit")
	}

	viper.SetEnvPrefix("NOMODIT")
	viper.AutomaticEnv()

	err = viper.ReadInConfig()
	if err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); ok {
			if home != "" {
				dir := filepath.Join(home, ".nomodit")
				if err := os.MkdirAll(dir, os.ModePerm); err == nil {
					if err := viper.WriteConfigAs(filepath.Join(dir, "config.env")); err != nil {

					}
				}

			}
			return nil
		}
		return err
	}
	return nil
}

func Save() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	dir := filepath.Join(home, ".nomodit")
	if err := viper.WriteConfigAs(filepath.Join(dir, "config.env")); err != nil {
		return err
	}
	return nil
}
