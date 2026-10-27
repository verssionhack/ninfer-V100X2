#include "serve/console_log.h"

#include <unistd.h>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace ninfer::serve {
namespace {

const char* level_name(ConsoleLogLevel level) noexcept {
    switch (level) {
    case ConsoleLogLevel::Info:
        return "info";
    case ConsoleLogLevel::Warning:
        return "warning";
    case ConsoleLogLevel::Error:
        return "error";
    }
    return "info";
}

std::mutex& console_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::string& status_text() {
    static std::string text;
    return text;
}

bool& status_open() {
    static bool open = false;
    return open;
}

std::size_t& status_width() {
    static std::size_t width = 0;
    return width;
}

// Callers hold console_mutex().
void erase_status_locked() {
    if (!status_open()) { return; }
    std::cerr << '\r' << std::string(status_width(), ' ') << '\r';
    status_open() = false;
}

// Callers hold console_mutex().
void draw_status_locked() {
    if (status_text().empty()) {
        status_width() = 0;
        return;
    }
    std::cerr << '\r' << status_text();
    status_width() = status_text().size();
    status_open()  = true;
}

} // namespace

std::string format_console_log_prefix(std::chrono::system_clock::time_point timestamp,
                                      ConsoleLogLevel level) {
    const auto since_epoch   = timestamp.time_since_epoch();
    const auto whole_seconds = std::chrono::duration_cast<std::chrono::seconds>(since_epoch);
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(since_epoch - whole_seconds).count();
    const std::time_t wall_seconds =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::time_point(whole_seconds));
    std::tm local{};
    localtime_r(&wall_seconds, &local);

    std::ostringstream out;
    out << '[' << std::put_time(&local, "%Y-%m-%d %H:%M:%S") << '.' << std::setfill('0')
        << std::setw(3) << milliseconds << "] [" << level_name(level) << "] ninfer-serve: ";
    return out.str();
}

std::string current_console_log_prefix(ConsoleLogLevel level) {
    return format_console_log_prefix(std::chrono::system_clock::now(), level);
}

void write_console_log(ConsoleLogLevel level, std::string_view message) {
    std::lock_guard lock(console_mutex());
    erase_status_locked();
    std::cerr << current_console_log_prefix(level) << message << '\n';
    draw_status_locked();
    std::cerr.flush();
}

void set_console_status_line(std::string_view line) {
    if (::isatty(STDERR_FILENO) != 1) { return; }
    std::lock_guard lock(console_mutex());
    status_text().assign(line);
    erase_status_locked();
    draw_status_locked();
    std::cerr.flush();
}

void clear_console_status_line() {
    std::lock_guard lock(console_mutex());
    erase_status_locked();
    status_text().clear();
    status_width() = 0;
    std::cerr.flush();
}

} // namespace ninfer::serve
