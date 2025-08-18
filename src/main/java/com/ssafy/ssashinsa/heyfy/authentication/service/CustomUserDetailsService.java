package com.ssafy.ssashinsa.heyfy.authentication.service;

import com.ssafy.ssashinsa.heyfy.user.domain.Users;
import com.ssafy.ssashinsa.heyfy.user.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.util.Collections;

@Service
@RequiredArgsConstructor
public class CustomUserDetailsService implements UserDetailsService {
    private final UserRepository userRepository;

    @Override
    public UserDetails loadUserByUsername(String studentId) throws UsernameNotFoundException {
        System.out.println("탐색 로직");
        // 현재 db에서 데이터 가져오도록 설정
        Users user = userRepository.findByStudentId(studentId).orElseThrow(() ->new UsernameNotFoundException("해당하는 사용자를 찾을 수 없습니다."));

        System.out.println("유저 불러옴 "+user.getUserId());
        return new User(
                user.getStudentId(),
                user.getPassword(),
                Collections.emptyList()
        );
    }
}