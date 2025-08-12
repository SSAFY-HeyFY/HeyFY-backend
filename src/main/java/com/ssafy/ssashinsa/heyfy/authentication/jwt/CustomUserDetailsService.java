package com.ssafy.ssashinsa.heyfy.authentication.config;

import com.ssafy.ssashinsa.heyfy.common.CustomException;
import com.ssafy.ssashinsa.heyfy.common.ErrorCode;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.Collections;

@Service
@RequiredArgsConstructor
public class CustomUserDetailsService implements UserDetailsService {

    private final PasswordEncoder passwordEncoder;

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        if (!"user".equals(username)) {
            throw new CustomException(ErrorCode.UNAUTHORIZED);
        }

        String encodedPassword = passwordEncoder.encode("ssafy");

        return new User(
                "user",
                encodedPassword,
                Collections.emptyList()
        );
    }
}