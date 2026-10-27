#pragma once

#include <chrono>
#include <string>
#include <string_view>

namespace ninfer::serve {

enum class ConsoleLogLevel {
    Info,
    Warning,
    Error,
};

[[nodiscard]] std::string format_console_log_prefix(std::chrono::system_clock::time_point timestamp,
                                                    ConsoleLogLevel level);

[[nodiscard]] std::string current_console_log_prefix(ConsoleLogLevel level);

void write_console_log(ConsoleLogLevel level, std::string_view message);

// Append a plain copy of every log line to `path`. The status line's in-place redraws are never
// written, so the file stays line-oriented even when stderr is an interactive terminal. Raw trace
// instrumentation (NINFER_REUSE_TRACE, NINFER_VISION_SWAP_TRACE) does not go through this logger.
// Returns false and fills `error` when the file cannot be opened.
[[nodiscard]] bool open_console_log_file(std::string_view path, std::string& error);

// Optional single in-place status line on stderr. It is only drawn when stderr is a terminal;
// `write_console_log` erases it before a log line and redraws it afterwards, all under the same
// mutex, so logs never interleave with the redraw. Passing an empty line clears it.
void set_console_status_line(std::string_view line);
void clear_console_status_line();

} // namespace ninfer::serve
